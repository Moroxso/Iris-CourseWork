"""End-to-end primitive lifecycle invariants for arm C."""

import torch

from iris.cognitive import ModuleProposal
from iris.shared_buffer import (
    SharedBufferLatentBridge,
    SharedBufferReader,
    SharedBufferState,
    SharedBufferWriter,
)


def _proposal(cycle: int) -> ModuleProposal:
    content = (
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + float(cycle)
    ).requires_grad_(True)
    return ModuleProposal(
        content=content,
        key=torch.full((1, 3, 4), float(cycle), requires_grad=True),
        bid=torch.full((1, 3), float(cycle), requires_grad=True),
        confidence=torch.full((1, 3), 0.5, requires_grad=True),
        ttl=torch.full((1, 3), 2.0, requires_grad=True),
    )


def test_three_cycles_use_three_pre_update_reads_plus_one_output_read() -> None:
    reader = SharedBufferReader(state_dim=4, num_modules=3, cycles=3)
    writer = SharedBufferWriter(state_dim=4, num_modules=3, cycles=3)
    bridge = SharedBufferLatentBridge(
        state_dim=4,
        backbone_hidden_size=6,
        num_modules=3,
        cycles=3,
    )
    state = SharedBufferState.empty(
        batch_size=1,
        state_dim=4,
        num_modules=3,
        cycles=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    goal = torch.tensor([[0.1, 0.2, -0.3, 0.4]], requires_grad=True)
    proposals = tuple(_proposal(cycle) for cycle in range(1, 4))
    pre_update_entry_counts: list[int] = []
    pre_update_attention_widths: list[int] = []
    post_write_entry_counts: list[int] = []

    for cycle_number, proposal in enumerate(proposals, start=1):
        pre_update_entry_counts.append(state.entry_count)
        pre_update_read = reader(state)
        pre_update_attention_widths.append(pre_update_read.attention_weights.shape[-1])
        if cycle_number == 1:
            assert torch.equal(
                pre_update_read.workspace_reads,
                torch.zeros_like(pre_update_read.workspace_reads),
            )
        state = writer(
            proposal,
            state,
            goal,
            cycle_number=cycle_number,
        )
        post_write_entry_counts.append(state.entry_count)

    final_output_read = reader(state)
    latent = bridge(final_output_read, torch.ones(1, 2, dtype=torch.bool))

    assert pre_update_entry_counts == [0, 3, 6]
    assert pre_update_attention_widths == [0, 3, 6]
    assert post_write_entry_counts == [3, 6, 9]
    assert final_output_read.attention_weights.shape == (1, 3, 9)
    assert latent.workspace_embeddings.shape == (1, 3, 6)

    weights = torch.linspace(0.1, 1.0, latent.workspace_embeddings.numel()).reshape_as(
        latent.workspace_embeddings
    )
    (latent.workspace_embeddings * weights).sum().backward()
    for proposal in proposals:
        assert proposal.content.grad is not None
        assert bool(torch.isfinite(proposal.content.grad).all())
        assert bool((proposal.content.grad != 0).any())
        assert proposal.key.grad is None
        assert proposal.bid.grad is None
        assert proposal.confidence.grad is None
        assert proposal.ttl.grad is None
    assert goal.grad is not None
    assert bool(torch.isfinite(goal.grad).all())
