"""Contract tests for functional gated writes into the global workspace."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch.nn import functional

from iris.cognitive import ModuleProposal
from iris.workspace import (
    SlotAllocatorOutput,
    WorkspaceState,
    WorkspaceWriter,
    WorkspaceWriterError,
)

CPU = torch.device("cpu")


def make_writer(
    *,
    state_dim: int = 4,
    num_modules: int = 3,
    num_slots: int = 4,
    winners_per_cycle: int = 2,
) -> WorkspaceWriter:
    """Create a compact writer with the production constructor contract."""
    return WorkspaceWriter(
        state_dim=state_dim,
        num_modules=num_modules,
        num_slots=num_slots,
        winners_per_cycle=winners_per_cycle,
    )


def make_proposal(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device = CPU,
    requires_grad: bool = False,
) -> ModuleProposal:
    """Build a finite proposal whose writer-consumed fields are non-degenerate."""
    generator = torch.Generator(device=device).manual_seed(104729)
    content = torch.randn(
        batch_size,
        num_modules,
        state_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    ).requires_grad_(requires_grad)
    scalar_shape = (batch_size, num_modules)
    confidence = (
        torch.linspace(0.2, 0.8, num_modules, dtype=dtype, device=device)
        .expand(batch_size, -1)
        .clone()
        .requires_grad_(requires_grad)
    )
    ttl = (
        torch.linspace(1.5, 3.5, num_modules, dtype=dtype, device=device)
        .expand(batch_size, -1)
        .clone()
        .requires_grad_(requires_grad)
    )
    return ModuleProposal(
        content=content,
        key=torch.zeros_like(content),
        bid=torch.zeros(scalar_shape, dtype=dtype, device=device),
        confidence=confidence,
        ttl=ttl,
    )


def make_workspace(
    *,
    batch_size: int = 2,
    num_slots: int = 4,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device = CPU,
    occupancy: torch.Tensor | None = None,
    requires_grad: bool = False,
    stale_empty_values: bool = False,
) -> WorkspaceState:
    """Build a valid state, optionally with stale metadata in empty slots."""
    if occupancy is None:
        occupancy = torch.tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype=torch.bool,
            device=device,
        )[:batch_size, :num_slots]
    generator = torch.Generator(device=device).manual_seed(130363)
    slots = torch.randn(
        batch_size,
        num_slots,
        state_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if not stale_empty_values:
        slots = slots * occupancy.unsqueeze(-1)
    slots.requires_grad_(requires_grad)
    age = torch.arange(num_slots, dtype=torch.int64, device=device).expand(batch_size, -1).clone()
    source = torch.arange(num_slots, dtype=torch.int64, device=device).remainder(3)
    source = source.expand(batch_size, -1).clone()
    confidence = torch.full((batch_size, num_slots), 0.4, dtype=dtype, device=device)
    ttl = occupancy.to(dtype=dtype) * 2.0
    write_count = (
        torch.arange(
            1,
            num_slots + 1,
            dtype=torch.int64,
            device=device,
        )
        .expand(batch_size, -1)
        .clone()
    )
    if not stale_empty_values:
        age = torch.where(occupancy, age, torch.zeros_like(age))
        source = torch.where(occupancy, source, torch.full_like(source, -1))
        confidence = torch.where(occupancy, confidence, torch.zeros_like(confidence))
        write_count = torch.where(occupancy, write_count, torch.zeros_like(write_count))
    return WorkspaceState(
        slots=slots,
        age=age,
        occupancy=occupancy,
        source_module=source,
        confidence=confidence,
        ttl=ttl,
        write_count=write_count,
    )


def make_allocation(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    num_slots: int = 4,
    winners_per_cycle: int = 2,
    dtype: torch.dtype = torch.float32,
    device: torch.device = CPU,
    module_indices: torch.Tensor | None = None,
    slot_indices: torch.Tensor | None = None,
    dense_surrogate: torch.Tensor | None = None,
) -> SlotAllocatorOutput:
    """Create unique hard assignments with an optional dense ST gradient path."""
    if module_indices is None:
        base_modules = torch.tensor([0, 2], dtype=torch.int64, device=device)
        module_indices = base_modules[:winners_per_cycle].expand(batch_size, -1).clone()
    if slot_indices is None:
        base_slots = torch.tensor([1, 3], dtype=torch.int64, device=device)
        slot_indices = base_slots[:winners_per_cycle].expand(batch_size, -1).clone()
    dense_shape = (batch_size, num_modules, num_slots)
    assignment_mask = torch.zeros(dense_shape, dtype=torch.bool, device=device)
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    assignment_mask[batch_indices, module_indices, slot_indices] = True
    target_mask = assignment_mask.any(dim=1)
    hard = assignment_mask.to(dtype=dtype)
    assignment_weights = hard
    if dense_surrogate is not None:
        assignment_weights = hard + (dense_surrogate - dense_surrogate.detach())

    row_mask = assignment_mask.any(dim=-1)
    target_grid = target_mask.unsqueeze(1).expand_as(assignment_mask)
    target_count = target_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=dtype)
    probabilities = target_grid.to(dtype=dtype) / target_count.unsqueeze(1)
    probabilities = probabilities * row_mask.unsqueeze(-1)
    return SlotAllocatorOutput(
        allocation_scores=torch.zeros(dense_shape, dtype=dtype, device=device),
        retention_scores=torch.zeros(batch_size, num_slots, dtype=dtype, device=device),
        target_mask=target_mask,
        assignment_mask=assignment_mask,
        assignment_probabilities=probabilities,
        assignment_weights=assignment_weights,
        module_indices=module_indices,
        slot_indices=slot_indices,
    )


def assert_canonical_empty(state: WorkspaceState) -> None:
    """Assert every field uses the canonical representation for empty slots."""
    empty = ~state.occupancy
    assert torch.equal(
        state.slots.masked_select(empty.unsqueeze(-1)),
        torch.zeros_like(state.slots.masked_select(empty.unsqueeze(-1))),
    )
    assert torch.equal(
        state.age.masked_select(empty), torch.zeros_like(state.age.masked_select(empty))
    )
    assert torch.equal(
        state.source_module.masked_select(empty),
        torch.full_like(state.source_module.masked_select(empty), -1),
    )
    assert torch.equal(
        state.confidence.masked_select(empty),
        torch.zeros_like(state.confidence.masked_select(empty)),
    )
    assert torch.equal(
        state.ttl.masked_select(empty), torch.zeros_like(state.ttl.masked_select(empty))
    )
    assert torch.equal(
        state.write_count.masked_select(empty),
        torch.zeros_like(state.write_count.masked_select(empty)),
    )


def test_matches_vector_gate_candidate_and_post_write_layer_norm_formula() -> None:
    writer = make_writer(state_dim=2, num_modules=2, num_slots=2, winners_per_cycle=1)
    with torch.no_grad():
        writer.gate_projection.weight.copy_(
            torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.5, 0.2], [-0.1, 0.6, 0.2, -0.4, 0.3, 0.5]])
        )
        writer.candidate_projection.weight.copy_(
            torch.tensor([[0.3, -0.7, 0.2, 0.4], [0.5, 0.1, -0.3, 0.6]])
        )
        writer.normalization.weight.copy_(torch.tensor([1.25, 0.75]))
        writer.normalization.bias.copy_(torch.tensor([0.1, -0.2]))

    proposal = ModuleProposal(
        content=torch.tensor([[[0.2, -0.4], [0.7, 0.1]]]),
        key=torch.zeros(1, 2, 2),
        bid=torch.zeros(1, 2),
        confidence=torch.tensor([[0.25, 0.9]]),
        ttl=torch.tensor([[2.0, 3.0]]),
    )
    workspace = WorkspaceState(
        slots=torch.tensor([[[0.6, -0.2], [0.0, 0.0]]]),
        age=torch.tensor([[4, 0]]),
        occupancy=torch.tensor([[True, False]]),
        source_module=torch.tensor([[0, -1]]),
        confidence=torch.tensor([[0.4, 0.0]]),
        ttl=torch.tensor([[2.0, 0.0]]),
        write_count=torch.tensor([[3, 0]]),
    )
    allocation = make_allocation(
        batch_size=1,
        num_modules=2,
        num_slots=2,
        winners_per_cycle=1,
        module_indices=torch.tensor([[1]]),
        slot_indices=torch.tensor([[0]]),
    )
    goal = torch.tensor([[0.3, -0.8]])

    output = writer(proposal, workspace, allocation, goal)
    old_slot = workspace.slots[0, 0]
    content = proposal.content[0, 1]
    gate = torch.sigmoid(
        functional.linear(torch.cat((content, old_slot, goal[0])), writer.gate_projection.weight)
    )
    candidate = torch.tanh(
        functional.linear(torch.cat((content, old_slot)), writer.candidate_projection.weight)
    )
    blended = (1.0 - gate) * old_slot + gate * candidate
    expected = functional.layer_norm(
        blended,
        (2,),
        writer.normalization.weight,
        writer.normalization.bias,
        writer.normalization.eps,
    )

    torch.testing.assert_close(output.slots[0, 0], expected)
    assert output.age.tolist() == [[0, 0]]
    assert output.occupancy.tolist() == [[True, False]]
    assert output.source_module.tolist() == [[1, -1]]
    torch.testing.assert_close(output.confidence, torch.tensor([[0.9, 0.0]]))
    torch.testing.assert_close(output.ttl, torch.tensor([[3.0, 0.0]]))
    assert output.write_count.tolist() == [[4, 0]]
    assert writer.gate_projection.bias is None
    assert writer.candidate_projection.bias is None
    assert writer.gate_projection.out_features == 2


def test_mixed_empty_and_occupied_batches_apply_functional_metadata_transitions() -> None:
    occupancy = torch.tensor(
        [[False, False, False, False], [True, False, True, False], [True, True, True, True]]
    )
    writer = make_writer()
    proposal = make_proposal(batch_size=3)
    workspace = make_workspace(
        batch_size=3,
        occupancy=occupancy,
        stale_empty_values=True,
    )
    allocation = make_allocation(batch_size=3)
    before = {name: getattr(workspace, name).clone() for name in workspace.__dataclass_fields__}

    output = writer(proposal, workspace, allocation, torch.full((3, 4), 0.15))
    selected = allocation.target_mask
    untouched_occupied = occupancy & ~selected

    assert torch.equal(output.occupancy, occupancy | selected)
    assert torch.equal(
        output.age.masked_select(selected), torch.zeros_like(output.age.masked_select(selected))
    )
    assert torch.equal(
        output.source_module.masked_select(selected),
        allocation.assignment_mask.to(torch.int64).argmax(dim=1).masked_select(selected),
    )
    expected_count = torch.where(
        occupancy, before["write_count"], torch.zeros_like(output.write_count)
    )
    expected_count = expected_count + selected.to(torch.int64)
    assert torch.equal(output.write_count, expected_count)
    assert torch.equal(
        output.slots.masked_select(untouched_occupied.unsqueeze(-1)),
        before["slots"].masked_select(untouched_occupied.unsqueeze(-1)),
    )
    assert torch.equal(
        output.age.masked_select(untouched_occupied),
        before["age"].masked_select(untouched_occupied),
    )
    for field_name in (
        "occupancy",
        "source_module",
        "confidence",
        "ttl",
        "write_count",
    ):
        assert torch.equal(
            getattr(output, field_name).masked_select(untouched_occupied),
            before[field_name].masked_select(untouched_occupied),
        )
    assert_canonical_empty(output)


def test_two_unique_writes_are_simultaneous_and_independent_of_rank_order() -> None:
    writer = make_writer()
    proposal = make_proposal()
    workspace = make_workspace()
    allocation = make_allocation()
    reverse = make_allocation(
        module_indices=allocation.module_indices.flip(1),
        slot_indices=allocation.slot_indices.flip(1),
    )

    first = writer(proposal, workspace, allocation, torch.zeros(2, 4))
    second = writer(proposal, workspace, reverse, torch.zeros(2, 4))

    for field_name in workspace.__dataclass_fields__:
        assert torch.equal(getattr(first, field_name), getattr(second, field_name))
    assert allocation.target_mask.sum(dim=1).tolist() == [2, 2]


def test_forward_does_not_mutate_any_input_tensor() -> None:
    writer = make_writer()
    proposal = make_proposal()
    workspace = make_workspace()
    allocation = make_allocation()
    goal = torch.randn(2, 4)
    objects = (proposal, workspace, allocation)
    snapshots = {
        (id(obj), name): value.clone()
        for obj in objects
        for name in obj.__dataclass_fields__
        if isinstance((value := getattr(obj, name)), torch.Tensor)
    }
    goal_snapshot = goal.clone()

    output = writer(proposal, workspace, allocation, goal)

    for obj in objects:
        for name in obj.__dataclass_fields__:
            value = getattr(obj, name)
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, snapshots[(id(obj), name)])
    assert torch.equal(goal, goal_snapshot)
    assert output is not workspace
    assert output.slots.data_ptr() != workspace.slots.data_ptr()


def test_dense_st_backward_reaches_writer_inputs_and_upstream_competition() -> None:
    writer = make_writer().to(dtype=torch.float64)
    proposal = make_proposal(dtype=torch.float64, requires_grad=True)
    workspace = make_workspace(dtype=torch.float64, requires_grad=True)
    goal = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    arbiter_scores = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    allocation_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    module_soft = torch.softmax(arbiter_scores, dim=-1).unsqueeze(-1)
    slot_soft = torch.softmax(allocation_logits, dim=-1)
    allocation = make_allocation(
        dtype=torch.float64,
        dense_surrogate=module_soft * slot_soft,
    )

    output = writer(proposal, workspace, allocation, goal)
    slot_coefficients = torch.linspace(0.2, 1.3, output.slots.numel(), dtype=torch.float64)
    loss = (output.slots.flatten() * slot_coefficients).sum()
    loss = loss + 0.7 * output.confidence.sum() + 0.3 * output.ttl.sum()
    loss.backward()

    for parameter in writer.parameters():
        assert parameter.grad is not None
        assert bool((parameter.grad != 0).any())
        assert bool(torch.isfinite(parameter.grad).all())
    for tensor in (proposal.content, proposal.confidence, proposal.ttl, workspace.slots, goal):
        assert tensor.grad is not None
        assert bool((tensor.grad != 0).any())
    assert arbiter_scores.grad is not None and bool((arbiter_scores.grad != 0).any())
    assert allocation_logits.grad is not None and bool((allocation_logits.grad != 0).any())


def test_two_functional_cycles_preserve_the_complete_training_graph() -> None:
    writer = make_writer().to(dtype=torch.float64)
    first_proposal = make_proposal(dtype=torch.float64, requires_grad=True)
    second_proposal = make_proposal(dtype=torch.float64, requires_grad=True)
    initial_workspace = make_workspace(dtype=torch.float64, requires_grad=True)
    first_goal = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    second_goal = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    first_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    second_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    first_allocation = make_allocation(
        dtype=torch.float64,
        dense_surrogate=torch.softmax(first_logits, dim=-1),
    )
    second_allocation = make_allocation(
        dtype=torch.float64,
        dense_surrogate=torch.softmax(second_logits, dim=-1),
    )

    first_state = writer(first_proposal, initial_workspace, first_allocation, first_goal)
    second_state = writer(second_proposal, first_state, second_allocation, second_goal)
    second_state.slots.square().sum().backward()

    for tensor in (
        initial_workspace.slots,
        first_proposal.content,
        second_proposal.content,
        first_goal,
        second_goal,
        first_logits,
        second_logits,
    ):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())
        assert bool((tensor.grad != 0).any())
    for parameter in writer.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_dense_st_gradient_uses_slot_level_layer_norm_after_combining_writes() -> None:
    writer = make_writer(
        state_dim=3,
        num_modules=2,
        num_slots=1,
        winners_per_cycle=1,
    ).to(dtype=torch.float64)
    proposal = make_proposal(
        batch_size=1,
        num_modules=2,
        state_dim=3,
        dtype=torch.float64,
    )
    workspace = make_workspace(
        batch_size=1,
        num_slots=1,
        state_dim=3,
        dtype=torch.float64,
        occupancy=torch.ones(1, 1, dtype=torch.bool),
    )
    goal = torch.tensor([[0.3, -0.4, 0.7]], dtype=torch.float64)
    dense_surrogate = torch.randn(1, 2, 1, dtype=torch.float64, requires_grad=True)
    allocation = make_allocation(
        batch_size=1,
        num_modules=2,
        num_slots=1,
        winners_per_cycle=1,
        dtype=torch.float64,
        module_indices=torch.tensor([[0]]),
        slot_indices=torch.tensor([[0]]),
        dense_surrogate=dense_surrogate,
    )

    output = writer(proposal, workspace, allocation, goal)
    content = proposal.content.unsqueeze(2)
    old_slot = workspace.slots.unsqueeze(1)
    expanded_goal = goal[:, None, None, :]
    gates = torch.sigmoid(
        functional.linear(
            torch.cat(
                (content, old_slot.expand(-1, 2, -1, -1), expanded_goal.expand(-1, 2, -1, -1)),
                dim=-1,
            ),
            writer.gate_projection.weight,
        )
    )
    candidates = torch.tanh(
        functional.linear(
            torch.cat((content, old_slot.expand(-1, 2, -1, -1)), dim=-1),
            writer.candidate_projection.weight,
        )
    )
    pair_updates = (1.0 - gates) * old_slot + gates * candidates
    assignments = allocation.assignment_weights
    combined = workspace.slots + torch.einsum(
        "bmk,bmkd->bkd",
        assignments,
        pair_updates - old_slot,
    )
    expected = functional.layer_norm(
        combined,
        (3,),
        writer.normalization.weight,
        writer.normalization.bias,
        writer.normalization.eps,
    )
    coefficients = torch.tensor([[[0.2, -0.7, 1.3]]], dtype=torch.float64)

    actual_gradient = torch.autograd.grad(
        (output.slots * coefficients).sum(),
        dense_surrogate,
        retain_graph=True,
    )[0]
    expected_gradient = torch.autograd.grad(
        (expected * coefficients).sum(),
        dense_surrogate,
        retain_graph=True,
    )[0]
    pairwise_normalized = functional.layer_norm(
        pair_updates,
        (3,),
        writer.normalization.weight,
        writer.normalization.bias,
        writer.normalization.eps,
    )
    legacy_pairwise = workspace.slots + torch.einsum(
        "bmk,bmkd->bkd",
        assignments,
        pairwise_normalized - old_slot,
    )
    legacy_gradient = torch.autograd.grad(
        (legacy_pairwise * coefficients).sum(),
        dense_surrogate,
    )[0]

    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(actual_gradient, legacy_gradient, rtol=1e-6, atol=1e-8)


def test_equal_metadata_has_zero_dense_selection_gradient() -> None:
    writer = make_writer().to(dtype=torch.float64)
    proposal = make_proposal(dtype=torch.float64)
    occupancy = torch.ones(2, 4, dtype=torch.bool)
    workspace = make_workspace(dtype=torch.float64, occupancy=occupancy)
    proposal = replace(
        proposal,
        confidence=torch.full_like(proposal.confidence, 0.4),
        ttl=torch.full_like(proposal.ttl, 2.0),
    )
    dense_surrogate = torch.randn(
        2,
        3,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    allocation = make_allocation(
        dtype=torch.float64,
        dense_surrogate=dense_surrogate,
    )

    output = writer(proposal, workspace, allocation, torch.zeros(2, 4, dtype=torch.float64))
    (output.confidence.sum() + output.ttl.sum()).backward()

    assert dense_surrogate.grad is not None
    assert torch.equal(dense_surrogate.grad, torch.zeros_like(dense_surrogate.grad))


def test_materializes_backbone_style_inference_tensors_for_trainable_backward() -> None:
    writer = make_writer()
    with torch.inference_mode():
        proposal = make_proposal()
        workspace = make_workspace()
        allocation = make_allocation()
        goal = torch.randn(2, 4)
    assert proposal.content.is_inference()
    assert workspace.slots.is_inference()

    output = writer(proposal, workspace, allocation, goal)
    assert not output.slots.is_inference()
    output.slots.square().sum().backward()
    assert writer.gate_projection.weight.grad is not None
    assert writer.candidate_projection.weight.grad is not None


def test_global_inference_mode_returns_valid_inference_state() -> None:
    writer = make_writer()
    with torch.inference_mode():
        output = writer(
            make_proposal(),
            make_workspace(),
            make_allocation(),
            torch.zeros(2, 4),
        )
    assert output.slots.is_inference()
    assert output.slots.shape == (2, 4, 4)
    assert bool(torch.isfinite(output.slots).all())
    assert_canonical_empty(output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_preserves_supported_floating_dtype_and_finite_output(dtype: torch.dtype) -> None:
    writer = make_writer().to(dtype=dtype)
    accumulation_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    output = writer(
        make_proposal(dtype=dtype),
        make_workspace(dtype=dtype),
        make_allocation(dtype=accumulation_dtype),
        torch.full((2, 4), 0.1, dtype=dtype),
    )

    assert output.slots.dtype == dtype
    assert output.confidence.dtype == dtype
    assert output.ttl.dtype == dtype
    assert bool(torch.isfinite(output.slots).all())
    assert_canonical_empty(output)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.float16, 60_000.0), (torch.bfloat16, 1.0e30)],
)
def test_low_precision_extreme_finite_inputs_accumulate_safely(
    dtype: torch.dtype, magnitude: float
) -> None:
    writer = make_writer().to(dtype=dtype)
    proposal = make_proposal(dtype=dtype)
    workspace = make_workspace(dtype=dtype)
    proposal = replace(
        proposal,
        content=torch.full_like(proposal.content, magnitude),
    )
    workspace = replace(
        workspace,
        slots=torch.where(
            workspace.occupancy.unsqueeze(-1),
            torch.full_like(workspace.slots, -magnitude),
            workspace.slots,
        ),
    )

    output = writer(
        proposal,
        workspace,
        make_allocation(dtype=torch.float32),
        torch.full((2, 4), magnitude, dtype=dtype),
    )

    assert output.slots.dtype == dtype
    assert bool(torch.isfinite(output.slots).all())


def test_cpu_autocast_cannot_silently_change_writer_contract_dtype() -> None:
    writer = make_writer()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = writer(
            make_proposal(),
            make_workspace(),
            make_allocation(),
            torch.zeros(2, 4),
        )
    assert output.slots.dtype == torch.float32
    assert bool(torch.isfinite(output.slots).all())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state_dim": 0}, "state_dim"),
        ({"num_modules": True}, "num_modules"),
        ({"num_slots": -1}, "num_slots"),
        ({"winners_per_cycle": 0}, "winners_per_cycle"),
        ({"num_modules": 1, "winners_per_cycle": 2}, "num_modules"),
        ({"num_slots": 1, "winners_per_cycle": 2}, "num_slots"),
    ],
)
def test_rejects_invalid_constructor_contract(kwargs: dict[str, Any], message: str) -> None:
    defaults: dict[str, Any] = {
        "state_dim": 4,
        "num_modules": 3,
        "num_slots": 4,
        "winners_per_cycle": 2,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        WorkspaceWriter(**defaults)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda proposal: cast(Any, "not a proposal"),
        lambda proposal: replace(proposal, content=proposal.content[:, :, :-1]),
        lambda proposal: replace(proposal, key=proposal.key.to(torch.float64)),
        lambda proposal: replace(
            proposal,
            bid=proposal.bid.clone().index_fill(1, torch.tensor([0]), float("nan")),
        ),
        lambda proposal: replace(proposal, confidence=torch.full_like(proposal.confidence, 1.1)),
        lambda proposal: replace(proposal, ttl=torch.zeros_like(proposal.ttl)),
    ],
)
def test_rejects_malformed_proposal(mutator: Any) -> None:
    writer = make_writer()
    with pytest.raises(WorkspaceWriterError):
        writer(
            mutator(make_proposal()),
            make_workspace(),
            make_allocation(),
            torch.zeros(2, 4),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: cast(Any, "not a workspace"),
        lambda state: replace(state, slots=state.slots[:, :, :-1]),
        lambda state: replace(state, occupancy=state.occupancy.to(torch.int64)),
        lambda state: replace(state, age=torch.full_like(state.age, -1)),
        lambda state: replace(state, source_module=torch.full_like(state.source_module, 99)),
        lambda state: replace(state, confidence=torch.full_like(state.confidence, float("inf"))),
        lambda state: replace(state, ttl=torch.full_like(state.ttl, -1.0)),
        lambda state: replace(
            state,
            ttl=torch.where(~state.occupancy, torch.ones_like(state.ttl), state.ttl),
        ),
    ],
)
def test_rejects_malformed_workspace(mutator: Any) -> None:
    writer = make_writer()
    with pytest.raises(WorkspaceWriterError):
        writer(
            make_proposal(),
            mutator(make_workspace()),
            make_allocation(),
            torch.zeros(2, 4),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda allocation: cast(Any, "not an allocation"),
        lambda allocation: replace(
            allocation,
            assignment_weights=allocation.assignment_weights + 0.1,
        ),
        lambda allocation: replace(allocation, target_mask=~allocation.target_mask),
        lambda allocation: replace(
            allocation,
            assignment_probabilities=torch.zeros_like(allocation.assignment_probabilities),
        ),
        lambda allocation: replace(
            allocation,
            module_indices=torch.full_like(allocation.module_indices, 99),
        ),
        lambda allocation: replace(
            allocation,
            module_indices=allocation.module_indices[:, :1].expand(-1, 2).clone(),
            slot_indices=allocation.slot_indices[:, :1].expand(-1, 2).clone(),
        ),
        lambda allocation: replace(
            allocation,
            allocation_scores=torch.full_like(allocation.allocation_scores, float("nan")),
        ),
    ],
)
def test_rejects_malformed_allocation(mutator: Any) -> None:
    writer = make_writer()
    with pytest.raises(WorkspaceWriterError):
        writer(
            make_proposal(),
            make_workspace(),
            mutator(make_allocation()),
            torch.zeros(2, 4),
        )


@pytest.mark.parametrize(
    "goal",
    [
        torch.zeros(2, 3),
        torch.zeros(2, 4, dtype=torch.float64),
        torch.full((2, 4), float("nan")),
        cast(Any, "not a tensor"),
    ],
)
def test_rejects_malformed_goal(goal: Any) -> None:
    with pytest.raises(WorkspaceWriterError):
        make_writer()(make_proposal(), make_workspace(), make_allocation(), goal)


def test_rejects_parameter_dtype_device_and_finite_boundary_violations() -> None:
    writer = make_writer().to(dtype=torch.float64)
    with pytest.raises(WorkspaceWriterError, match="parameters must share"):
        writer(make_proposal(), make_workspace(), make_allocation(), torch.zeros(2, 4))

    writer = make_writer()
    with torch.no_grad():
        writer.gate_projection.weight[0, 0] = float("nan")
    with pytest.raises(WorkspaceWriterError, match=r"parameter .* contains NaN or Inf"):
        writer(make_proposal(), make_workspace(), make_allocation(), torch.zeros(2, 4))


def test_output_has_complete_shape_dtype_device_finiteness_and_metadata_invariants() -> None:
    writer = make_writer()
    output = writer(
        make_proposal(),
        make_workspace(),
        make_allocation(),
        torch.zeros(2, 4),
    )

    assert output.slots.shape == (2, 4, 4)
    for field_name in ("age", "occupancy", "source_module", "confidence", "ttl", "write_count"):
        assert getattr(output, field_name).shape == (2, 4)
        assert getattr(output, field_name).device == output.slots.device
    assert output.slots.dtype == torch.float32
    assert output.age.dtype == torch.int64
    assert output.occupancy.dtype == torch.bool
    assert output.source_module.dtype == torch.int64
    assert output.confidence.dtype == torch.float32
    assert output.ttl.dtype == torch.float32
    assert output.write_count.dtype == torch.int64
    assert bool(torch.isfinite(output.slots).all())
    assert bool(torch.logical_and(output.confidence >= 0, output.confidence <= 1).all())
    assert bool((output.ttl >= 0).all())
    assert bool((output.age >= 0).all())
    assert bool((output.write_count >= 0).all())
    assert bool(torch.logical_and(output.source_module >= -1, output.source_module < 3).all())
    assert_canonical_empty(output)
