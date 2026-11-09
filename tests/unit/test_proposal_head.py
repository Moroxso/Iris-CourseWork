"""Contract tests for latent cognitive-module proposal heads."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional

from iris.cognitive import (
    CognitiveModule,
    ModuleProposal,
    ProposalHead,
    ProposalHeadError,
)


def proposal_states(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create deterministic, non-degenerate module states."""
    generator = torch.Generator().manual_seed(104729)
    return torch.randn(
        batch_size,
        num_modules,
        state_dim,
        generator=generator,
        dtype=dtype,
    )


def backward_through_all_fields(proposal: ModuleProposal) -> None:
    """Backpropagate one scalar objective that uses every proposal field."""
    loss = (
        proposal.content.float().square().mean()
        + proposal.key.float().square().mean()
        + proposal.bid.float().square().mean()
        + proposal.confidence.float().square().mean()
        + proposal.ttl.float().square().mean()
    )
    torch.autograd.backward(loss)


def test_has_exactly_five_shared_bias_free_projection_heads() -> None:
    model = ProposalHead(state_dim=4, num_modules=8, cycles=3)

    heads = {
        "content_head": (4, 4),
        "key_head": (4, 4),
        "bid_head": (4, 1),
        "confidence_head": (4, 1),
        "ttl_head": (4, 1),
    }
    assert set(dict(model.named_children())) == set(heads)
    for name, (in_features, out_features) in heads.items():
        head = cast(nn.Linear, getattr(model, name))
        assert isinstance(head, nn.Linear)
        assert (head.in_features, head.out_features) == (in_features, out_features)
        assert head.bias is None


def test_matches_complete_manual_proposal_equations() -> None:
    torch.manual_seed(101)
    model = ProposalHead(state_dim=3, num_modules=2, cycles=5).to(dtype=torch.float64)
    states = proposal_states(
        batch_size=2,
        num_modules=2,
        state_dim=3,
        dtype=torch.float64,
    )

    actual = model(states)
    expected_content = functional.linear(states, model.content_head.weight)
    expected_key = functional.linear(states, model.key_head.weight)
    expected_bid = functional.linear(states, model.bid_head.weight).squeeze(-1)
    expected_confidence = torch.sigmoid(
        functional.linear(states, model.confidence_head.weight).squeeze(-1)
    )
    expected_ttl = 1.0 + 4.0 * torch.sigmoid(
        functional.linear(states, model.ttl_head.weight).squeeze(-1)
    )

    torch.testing.assert_close(actual.content, expected_content, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.key, expected_key, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.bid, expected_bid, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.confidence, expected_confidence, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual.ttl, expected_ttl, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("batch_size", "num_modules"),
    [(1, 1), (1, 3), (2, 1)],
)
def test_returns_typed_frozen_output_with_scalar_and_vector_shapes(
    batch_size: int,
    num_modules: int,
) -> None:
    model = ProposalHead(state_dim=4, num_modules=num_modules, cycles=3)
    states = proposal_states(batch_size=batch_size, num_modules=num_modules)

    output = model(states)

    assert isinstance(output, ModuleProposal)
    assert output.content.shape == (batch_size, num_modules, 4)
    assert output.key.shape == (batch_size, num_modules, 4)
    assert output.bid.shape == (batch_size, num_modules)
    assert output.confidence.shape == (batch_size, num_modules)
    assert output.ttl.shape == (batch_size, num_modules)
    for tensor in (
        output.content,
        output.key,
        output.bid,
        output.confidence,
        output.ttl,
    ):
        assert tensor.dtype == states.dtype
        assert tensor.device == states.device
        assert torch.isfinite(tensor).all()
    with pytest.raises(FrozenInstanceError):
        output.bid = torch.zeros_like(output.bid)  # type: ignore[misc]


def test_zero_state_has_exact_neutral_proposal() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=5)

    output = model(torch.zeros(2, 3, 4))

    assert torch.count_nonzero(output.content) == 0
    assert torch.count_nonzero(output.key) == 0
    assert torch.count_nonzero(output.bid) == 0
    torch.testing.assert_close(output.confidence, torch.full((2, 3), 0.5), rtol=0, atol=0)
    torch.testing.assert_close(output.ttl, torch.full((2, 3), 3.0), rtol=0, atol=0)


def test_ttl_is_fractional_and_is_not_rounded() -> None:
    model = ProposalHead(state_dim=1, num_modules=1, cycles=4)
    with torch.no_grad():
        model.ttl_head.weight.fill_(1.0)
    states = torch.tensor([[[0.4]]])

    ttl = model(states).ttl
    expected = 1.0 + 3.0 * torch.sigmoid(torch.tensor(0.4))

    torch.testing.assert_close(ttl.squeeze(), expected)
    assert not torch.equal(ttl, ttl.round())


def test_single_cycle_ttl_is_exactly_one_for_every_finite_logit() -> None:
    model = ProposalHead(state_dim=2, num_modules=3, cycles=1)
    with torch.no_grad():
        model.ttl_head.weight.copy_(torch.tensor([[13.0, -17.0]]))
    states = torch.tensor([[[1.0, 1.0], [-1.0, -1.0], [10.0, -10.0]]])

    ttl = model(states).ttl

    torch.testing.assert_close(ttl, torch.ones_like(ttl), rtol=0, atol=0)


def test_shared_heads_are_module_permutation_equivariant() -> None:
    model = ProposalHead(state_dim=4, num_modules=4, cycles=3)
    states = proposal_states(num_modules=4)
    permutation = torch.tensor([2, 0, 3, 1])
    reference = model(states)

    permuted = model(states[:, permutation])

    for actual, expected in (
        (permuted.content, reference.content[:, permutation]),
        (permuted.key, reference.key[:, permutation]),
        (permuted.bid, reference.bid[:, permutation]),
        (permuted.confidence, reference.confidence[:, permutation]),
        (permuted.ttl, reference.ttl[:, permutation]),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_changing_one_module_only_changes_its_own_proposal() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    states = proposal_states()
    reference = model(states)
    changed_states = states.detach().clone()
    changed_states[:, 1] += torch.tensor([2.0, -3.0, 5.0, -7.0])

    changed = model(changed_states)

    for field in ("content", "key", "bid", "confidence", "ttl"):
        reference_tensor = cast(torch.Tensor, getattr(reference, field))
        changed_tensor = cast(torch.Tensor, getattr(changed, field))
        torch.testing.assert_close(changed_tensor[:, 0], reference_tensor[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(changed_tensor[:, 2], reference_tensor[:, 2], rtol=0, atol=0)
        assert not torch.equal(changed_tensor[:, 1], reference_tensor[:, 1])


def test_bid_is_an_unbounded_raw_score() -> None:
    model = ProposalHead(state_dim=2, num_modules=2, cycles=3)
    with torch.no_grad():
        model.bid_head.weight.copy_(torch.tensor([[2.0, 0.0]]))
    states = torch.tensor([[[2.0, 0.0], [-3.0, 0.0]]])

    bid = model(states).bid

    torch.testing.assert_close(bid, torch.tensor([[4.0, -6.0]]), rtol=0, atol=0)
    assert bid.max() > 1
    assert bid.min() < 0


def test_all_proposal_fields_backpropagate_to_states_and_every_head() -> None:
    torch.manual_seed(211)
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    states = proposal_states().requires_grad_()

    backward_through_all_fields(model(states))

    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    assert torch.count_nonzero(states.grad) > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_integrates_with_cognitive_module_and_backpropagates_end_to_end() -> None:
    torch.manual_seed(307)
    cognitive = CognitiveModule(state_dim=4, num_modules=3)
    proposal_head = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    generator = torch.Generator().manual_seed(311)
    prompt_context = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
    workspace_reads = torch.randn(2, 3, 4, generator=generator, requires_grad=True)
    goal_state = torch.randn(2, 4, generator=generator, requires_grad=True)

    first_cycle = cognitive(prompt_context, workspace_reads, goal_state).module_states
    module_states = cognitive(
        prompt_context,
        workspace_reads,
        goal_state,
        first_cycle,
    ).module_states
    module_states.retain_grad()
    backward_through_all_fields(proposal_head(module_states))

    assert module_states.grad is not None
    assert torch.isfinite(module_states.grad).all()
    assert torch.count_nonzero(module_states.grad) > 0
    for tensor in (prompt_context, workspace_reads, goal_state):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    for parameter in (*cognitive.parameters(), *proposal_head.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_materializes_inference_input_for_trainable_backward() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    with torch.inference_mode():
        states = proposal_states()

    output = model(states)
    backward_through_all_fields(output)

    assert not output.content.is_inference()
    assert output.content.requires_grad
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_supports_global_inference_mode() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)

    with torch.inference_mode():
        output = model(proposal_states())

    for tensor in (
        output.content,
        output.key,
        output.bid,
        output.confidence,
        output.ttl,
    ):
        assert tensor.is_inference()
        assert not tensor.requires_grad
        assert torch.isfinite(tensor).all()


def test_float64_preserves_dtype_and_exact_manual_precision() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3).to(dtype=torch.float64)
    states = proposal_states(dtype=torch.float64)

    output = model(states)

    for tensor in (
        output.content,
        output.key,
        output.bid,
        output.confidence,
        output.ttl,
    ):
        assert tensor.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_reduced_precision_forward_and_backward_are_finite(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(401)
    model = ProposalHead(state_dim=8, num_modules=3, cycles=3).to(dtype=dtype)
    states = proposal_states(state_dim=8, dtype=dtype).requires_grad_()

    output = model(states)
    backward_through_all_fields(output)

    for tensor in (
        output.content,
        output.key,
        output.bid,
        output.confidence,
        output.ttl,
    ):
        assert tensor.dtype == dtype
        assert torch.isfinite(tensor).all()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_fp16_bounded_heads_accumulate_logits_in_float32() -> None:
    model = ProposalHead(state_dim=8, num_modules=1, cycles=3).to(dtype=torch.float16)
    with torch.no_grad():
        for child in model.children():
            cast(nn.Linear, child).weight.zero_()
        model.confidence_head.weight.fill_(0.25)
        model.ttl_head.weight.fill_(0.25)
    states = torch.full((1, 1, 8), 60_000.0, dtype=torch.float16)

    output = model(states)

    assert torch.count_nonzero(output.content) == 0
    assert torch.count_nonzero(output.key) == 0
    assert torch.count_nonzero(output.bid) == 0
    torch.testing.assert_close(output.confidence, torch.ones_like(output.confidence))
    torch.testing.assert_close(output.ttl, torch.full_like(output.ttl, 3.0))


def test_bfloat16_cancelling_logits_accumulate_in_float32() -> None:
    model = ProposalHead(state_dim=8, num_modules=1, cycles=3).to(dtype=torch.bfloat16)
    cancelling_row = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for child in model.children():
            head = cast(nn.Linear, child)
            head.weight.copy_(cancelling_row.expand_as(head.weight))
    states = torch.full((1, 1, 8), 1e38, dtype=torch.bfloat16)

    output = model(states)

    assert torch.count_nonzero(output.content) == 0
    assert torch.count_nonzero(output.key) == 0
    assert torch.count_nonzero(output.bid) == 0
    torch.testing.assert_close(
        output.confidence,
        torch.full_like(output.confidence, 0.5),
    )
    torch.testing.assert_close(output.ttl, torch.full_like(output.ttl, 2.0))


def test_rejects_output_that_overflows_when_cast_to_input_dtype() -> None:
    model = ProposalHead(state_dim=2, num_modules=1, cycles=3).to(dtype=torch.float16)
    with torch.no_grad():
        for child in model.children():
            cast(nn.Linear, child).weight.zero_()
        model.content_head.weight[0, 0] = 2.0
    states = torch.tensor([[[40_000.0, 0.0]]], dtype=torch.float16)

    with pytest.raises(
        ProposalHeadError,
        match="Proposal content contains NaN or Inf",
    ):
        model(states)


def test_external_autocast_cannot_lower_proposal_precision() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    states = proposal_states()
    reference = model(states)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model(states)

    for actual_tensor, reference_tensor in (
        (actual.content, reference.content),
        (actual.key, reference.key),
        (actual.bid, reference.bid),
        (actual.confidence, reference.confidence),
        (actual.ttl, reference.ttl),
    ):
        assert actual_tensor.dtype == torch.float32
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("state_dim", "num_modules", "cycles"),
    [
        (0, 2, 3),
        (-1, 2, 3),
        (True, 2, 3),
        (2.5, 2, 3),
        (2, 0, 3),
        (2, -1, 3),
        (2, True, 3),
        (2, 2.5, 3),
        (2, 2, 0),
        (2, 2, -1),
        (2, 2, True),
        (2, 2, 2.5),
    ],
)
def test_rejects_invalid_constructor_dimensions(
    state_dim: object,
    num_modules: object,
    cycles: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ProposalHead(
            state_dim=state_dim,  # type: ignore[arg-type]
            num_modules=num_modules,  # type: ignore[arg-type]
            cycles=cycles,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "replacement",
    [
        "not-a-tensor",
        torch.ones(2, 3),
        torch.ones(0, 3, 4),
        torch.ones(2, 0, 4),
        torch.ones(2, 2, 4),
        torch.ones(2, 3, 5),
        torch.ones(2, 3, 4, dtype=torch.int64),
        torch.full((2, 3, 4), torch.nan),
        torch.full((2, 3, 4), torch.inf),
    ],
    ids=[
        "nontensor",
        "rank",
        "empty-batch",
        "empty-modules",
        "module-count",
        "width",
        "integer",
        "nan",
        "inf",
    ],
)
def test_rejects_malformed_module_states(replacement: object) -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)

    with pytest.raises(ProposalHeadError):
        model(cast(Any, replacement))


def test_rejects_parameter_dtype_mismatch() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3).to(dtype=torch.float64)

    with pytest.raises(ProposalHeadError, match="parameters must explicitly match"):
        model(proposal_states())


def test_rejects_parameter_device_mismatch() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3).to(device=torch.device("meta"))

    with pytest.raises(ProposalHeadError, match="parameters must explicitly match"):
        model(proposal_states())


def test_rejects_nonfinite_parameter() -> None:
    model = ProposalHead(state_dim=4, num_modules=3, cycles=3)
    with torch.no_grad():
        model.key_head.weight[0, 0] = torch.nan

    with pytest.raises(ProposalHeadError, match=r"parameter .* contains NaN or Inf"):
        model(proposal_states())


@pytest.mark.parametrize(
    ("head_name", "field_name"),
    [
        ("content_head", "content"),
        ("key_head", "key"),
        ("bid_head", "bid"),
        ("confidence_head", "confidence"),
        ("ttl_head", "ttl"),
    ],
)
def test_rejects_raw_projection_overflow_before_nonlinearities(
    head_name: str,
    field_name: str,
) -> None:
    model = ProposalHead(state_dim=2, num_modules=1, cycles=3)
    with torch.no_grad():
        for child in model.children():
            cast(nn.Linear, child).weight.zero_()
        target = cast(nn.Linear, getattr(model, head_name))
        target.weight[0, 0] = 2.0
    states = torch.zeros(1, 1, 2)
    states[0, 0, 0] = torch.finfo(torch.float32).max

    with pytest.raises(
        ProposalHeadError,
        match=rf"Proposal {field_name} logits contain NaN or Inf",
    ):
        model(states)
