"""Contract tests for the independent recurrent cognitive-module bank."""

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from iris.cognitive import CognitiveModule, CognitiveModuleError, CognitiveModuleOutput


def module_inputs(
    *,
    batch_size: int = 2,
    num_modules: int = 3,
    state_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create finite, non-degenerate inputs for one cognitive cycle."""
    generator = torch.Generator().manual_seed(1729)
    prompt_context = torch.randn(
        batch_size, num_modules, state_dim, generator=generator, dtype=dtype
    )
    workspace_reads = torch.randn(
        batch_size, num_modules, state_dim, generator=generator, dtype=dtype
    )
    goal_state = torch.randn(batch_size, state_dim, generator=generator, dtype=dtype)
    return prompt_context, workspace_reads, goal_state


def test_exact_architecture_and_fused_input_order() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    call_order: list[str] = []
    norm_inputs: list[torch.Tensor] = []

    for name in (
        "input_norm",
        "input_projection",
        "activation",
        "recurrent_cell",
        "output_projection",
    ):
        submodule = cast(nn.Module, getattr(model.module_cells[0], name))
        submodule.register_forward_pre_hook(
            lambda _module, inputs, name=name: call_order.append(name)
        )
    cast(nn.Module, model.module_cells[0].input_norm).register_forward_pre_hook(
        lambda _module, inputs: norm_inputs.append(inputs[0].detach().clone())
    )

    output = model(prompt_context, workspace_reads, goal_state)
    first_cell = cast(Any, model.module_cells[0])

    assert call_order == [
        "input_norm",
        "input_projection",
        "activation",
        "recurrent_cell",
        "output_projection",
    ]
    assert isinstance(first_cell.input_norm, nn.LayerNorm)
    assert first_cell.input_norm.normalized_shape == (16,)
    assert isinstance(first_cell.input_projection, nn.Linear)
    assert (first_cell.input_projection.in_features, first_cell.input_projection.out_features) == (
        16,
        4,
    )
    assert isinstance(first_cell.activation, nn.SiLU)
    assert isinstance(first_cell.recurrent_cell, nn.GRUCell)
    assert (first_cell.recurrent_cell.input_size, first_cell.recurrent_cell.hidden_size) == (4, 4)
    assert isinstance(first_cell.output_projection, nn.Linear)
    assert (
        first_cell.output_projection.in_features,
        first_cell.output_projection.out_features,
    ) == (
        4,
        4,
    )
    expected_fused = torch.cat(
        (
            prompt_context[:, 0],
            workspace_reads[:, 0],
            goal_state,
            first_cell.identity_embedding.unsqueeze(0).expand(prompt_context.shape[0], -1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(norm_inputs[0], expected_fused)
    assert output.module_states.shape == (2, 3, 4)


def test_cell_matches_the_complete_manual_recurrent_equation() -> None:
    torch.manual_seed(101)
    model = CognitiveModule(state_dim=3, num_modules=1).to(dtype=torch.float64)
    prompt_context, workspace_reads, goal_state = module_inputs(
        batch_size=2,
        num_modules=1,
        state_dim=3,
        dtype=torch.float64,
    )
    previous_states = torch.randn(2, 1, 3, dtype=torch.float64)
    cell = cast(Any, model.module_cells[0])

    fused = torch.cat(
        (
            prompt_context[:, 0],
            workspace_reads[:, 0],
            goal_state,
            cell.identity_embedding.unsqueeze(0).expand(2, -1),
        ),
        dim=-1,
    )
    normalized = functional.layer_norm(
        fused,
        (12,),
        cell.input_norm.weight,
        cell.input_norm.bias,
        cell.input_norm.eps,
    )
    prepared = functional.silu(
        functional.linear(
            normalized,
            cell.input_projection.weight,
            cell.input_projection.bias,
        )
    )
    input_gates = functional.linear(
        prepared,
        cell.recurrent_cell.weight_ih,
        cell.recurrent_cell.bias_ih,
    )
    hidden_gates = functional.linear(
        previous_states[:, 0],
        cell.recurrent_cell.weight_hh,
        cell.recurrent_cell.bias_hh,
    )
    input_reset, input_update, input_candidate = input_gates.chunk(3, dim=-1)
    hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=-1)
    reset = torch.sigmoid(input_reset + hidden_reset)
    update = torch.sigmoid(input_update + hidden_update)
    candidate = torch.tanh(input_candidate + reset * hidden_candidate)
    recurrent_state = (1 - update) * candidate + update * previous_states[:, 0]
    expected = functional.linear(
        recurrent_state,
        cell.output_projection.weight,
        cell.output_projection.bias,
    )

    actual = model(
        prompt_context,
        workspace_reads,
        goal_state,
        previous_states,
    ).module_states[:, 0]

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_returns_typed_frozen_output_and_preserves_boundary() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs(dtype=torch.float64)
    model = model.to(dtype=torch.float64)

    output = model(prompt_context, workspace_reads, goal_state)

    assert isinstance(output, CognitiveModuleOutput)
    assert output.module_states.shape == (2, 3, 4)
    assert output.module_states.dtype == torch.float64
    assert output.module_states.device == prompt_context.device
    assert torch.isfinite(output.module_states).all()
    with pytest.raises(FrozenInstanceError):
        output.module_states = torch.zeros_like(output.module_states)  # type: ignore[misc]


def test_default_previous_state_is_exactly_explicit_zero_state() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    zeros = model.initial_state(
        batch_size=prompt_context.shape[0],
        dtype=prompt_context.dtype,
        device=prompt_context.device,
    )

    implicit = model(prompt_context, workspace_reads, goal_state)
    explicit = model(prompt_context, workspace_reads, goal_state, zeros)

    assert torch.count_nonzero(zeros) == 0
    torch.testing.assert_close(implicit.module_states, explicit.module_states, rtol=0, atol=0)


def test_recurrence_receives_projected_previous_cycle_state() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    first_cycle = model(prompt_context, workspace_reads, goal_state)
    received_previous_states: list[torch.Tensor] = []
    for raw_cell in model.module_cells:
        cell = cast(Any, raw_cell)
        cast(nn.Module, cell.recurrent_cell).register_forward_pre_hook(
            lambda _module, inputs: received_previous_states.append(inputs[1].detach().clone())
        )

    second_cycle = model(
        prompt_context,
        workspace_reads,
        goal_state,
        first_cycle.module_states,
    )
    restarted = model(prompt_context, workspace_reads, goal_state)

    assert len(received_previous_states) == 6
    for module_index in range(model.num_modules):
        torch.testing.assert_close(
            received_previous_states[module_index],
            first_cycle.module_states[:, module_index],
            rtol=0,
            atol=0,
        )
    assert not torch.equal(second_cycle.module_states, restarted.module_states)


def test_cells_have_disjoint_parameters_and_learned_identities() -> None:
    model = CognitiveModule(state_dim=4, num_modules=8)
    parameter_ids = [
        {id(parameter) for parameter in cell.parameters()} for cell in model.module_cells
    ]
    storage_pointers = [
        {parameter.untyped_storage().data_ptr() for parameter in cell.parameters()}
        for cell in model.module_cells
    ]

    for left_index in range(model.num_modules):
        for right_index in range(left_index + 1, model.num_modules):
            assert parameter_ids[left_index].isdisjoint(parameter_ids[right_index])
            assert storage_pointers[left_index].isdisjoint(storage_pointers[right_index])
    identities = [
        cast(torch.Tensor, cast(Any, cell).identity_embedding) for cell in model.module_cells
    ]
    assert all(
        isinstance(identity, nn.Parameter) and identity.requires_grad for identity in identities
    )
    assert all(identity.shape == (4,) for identity in identities)
    assert any(not torch.equal(identities[0], identity) for identity in identities[1:])


def test_changing_one_identity_only_changes_that_module() -> None:
    torch.manual_seed(29)
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    baseline = model(prompt_context, workspace_reads, goal_state).module_states.detach()

    with torch.no_grad():
        identity = cast(torch.Tensor, cast(Any, model.module_cells[1]).identity_embedding)
        identity.add_(torch.tensor([2.0, -3.0, 5.0, -7.0]))
    changed = model(prompt_context, workspace_reads, goal_state).module_states.detach()

    torch.testing.assert_close(changed[:, 0], baseline[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(changed[:, 2], baseline[:, 2], rtol=0, atol=0)
    assert not torch.equal(changed[:, 1], baseline[:, 1])


def test_backward_crosses_two_cycles_and_reaches_every_cell_and_input() -> None:
    torch.manual_seed(43)
    model = CognitiveModule(state_dim=4, num_modules=3)
    tensors = module_inputs()
    prompt_context, workspace_reads, goal_state = (
        tensor.detach().clone().requires_grad_() for tensor in tensors
    )

    first_cycle = model(prompt_context, workspace_reads, goal_state)
    first_cycle.module_states.retain_grad()
    second_cycle = model(
        prompt_context,
        workspace_reads,
        goal_state,
        first_cycle.module_states,
    )
    second_cycle.module_states.square().sum().backward()

    assert first_cycle.module_states.grad is not None
    assert torch.isfinite(first_cycle.module_states.grad).all()
    assert torch.count_nonzero(first_cycle.module_states.grad) > 0
    for tensor in (prompt_context, workspace_reads, goal_state):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_loss_on_one_module_does_not_reach_other_cells() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()

    output = model(prompt_context, workspace_reads, goal_state)
    output.module_states[:, 1].square().sum().backward()

    for module_index, cell in enumerate(model.module_cells):
        gradients = [parameter.grad for parameter in cell.parameters()]
        for parameter in cell.parameters():
            if module_index == 1:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all()
            else:
                assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        if module_index == 1:
            assert any(
                gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients
            )


def test_materializes_inference_inputs_for_trainable_backward() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    with torch.inference_mode():
        prompt_context, workspace_reads, goal_state = module_inputs()
        previous_states = torch.randn(2, 3, 4)

    output = model(prompt_context, workspace_reads, goal_state, previous_states)
    output.module_states.square().sum().backward()

    assert not output.module_states.is_inference()
    assert output.module_states.requires_grad
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_supports_trainable_parameters_under_global_inference_mode() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)

    with torch.inference_mode():
        prompt_context, workspace_reads, goal_state = module_inputs()
        output = model(prompt_context, workspace_reads, goal_state)

    assert output.module_states.is_inference()
    assert not output.module_states.requires_grad
    assert torch.isfinite(output.module_states).all()


def test_external_autocast_cannot_lower_module_precision() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    reference = model(prompt_context, workspace_reads, goal_state)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = model(prompt_context, workspace_reads, goal_state)

    assert actual.module_states.dtype == torch.float32
    torch.testing.assert_close(actual.module_states, reference.module_states, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_reduced_precision_two_cycle_backward_is_finite(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(73)
    model = CognitiveModule(state_dim=8, num_modules=3).to(dtype=dtype)
    prompt_context, workspace_reads, goal_state = module_inputs(
        num_modules=3,
        state_dim=8,
        dtype=dtype,
    )

    first = model(prompt_context, workspace_reads, goal_state)
    second = model(
        prompt_context,
        workspace_reads,
        goal_state,
        first.module_states,
    )
    second.module_states.float().square().mean().backward()

    assert torch.isfinite(first.module_states).all()
    assert torch.isfinite(second.module_states).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("state_dim", "num_modules"),
    [(0, 2), (-1, 2), (True, 2), (2.5, 2), (2, 0), (2, -1), (2, True), (2, 2.5)],
)
def test_rejects_invalid_constructor_dimensions(state_dim: object, num_modules: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CognitiveModule(
            state_dim=state_dim,  # type: ignore[arg-type]
            num_modules=num_modules,  # type: ignore[arg-type]
        )


def test_initial_state_preserves_explicit_boundary() -> None:
    model = CognitiveModule(state_dim=3, num_modules=2)

    state = model.initial_state(
        batch_size=4,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert state.shape == (4, 2, 3)
    assert state.dtype == torch.float64
    assert state.device == torch.device("cpu")
    assert torch.count_nonzero(state) == 0


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_initial_state_rejects_invalid_batch_size(batch_size: object) -> None:
    model = CognitiveModule(state_dim=3, num_modules=2)

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        model.initial_state(
            batch_size=batch_size,  # type: ignore[arg-type]
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("dtype", "device", "match"),
    [
        (None, torch.device("cpu"), "explicit torch.dtype"),
        ("float32", torch.device("cpu"), "explicit torch.dtype"),
        (torch.float32, None, "explicit torch.device"),
        (torch.float32, "cpu", "explicit torch.device"),
        (torch.int64, torch.device("cpu"), "floating point"),
    ],
)
def test_initial_state_rejects_invalid_dtype_or_device(
    dtype: object, device: object, match: str
) -> None:
    model = CognitiveModule(state_dim=3, num_modules=2)

    with pytest.raises(ValueError, match=match):
        model.initial_state(
            batch_size=1,
            dtype=dtype,  # type: ignore[arg-type]
            device=device,  # type: ignore[arg-type]
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
def test_rejects_malformed_module_inputs(replacement: object) -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    _, workspace_reads, goal_state = module_inputs()

    with pytest.raises(CognitiveModuleError):
        model(replacement, workspace_reads, goal_state)


@pytest.mark.parametrize(
    ("argument", "replacement"),
    [
        ("workspace_reads", "not-a-tensor"),
        ("workspace_reads", torch.ones(2, 3)),
        ("workspace_reads", torch.ones(2, 2, 4)),
        ("workspace_reads", torch.ones(2, 3, 5)),
        ("workspace_reads", torch.ones(2, 3, 4, dtype=torch.int64)),
        ("workspace_reads", torch.full((2, 3, 4), torch.nan)),
        ("goal_state", "not-a-tensor"),
        ("goal_state", torch.ones(2, 1, 4)),
        ("goal_state", torch.ones(1, 4)),
        ("goal_state", torch.ones(2, 5)),
        ("goal_state", torch.ones(2, 4, dtype=torch.int64)),
        ("goal_state", torch.full((2, 4), torch.inf)),
        ("previous_states", "not-a-tensor"),
        ("previous_states", torch.ones(2, 3)),
        ("previous_states", torch.ones(2, 2, 4)),
        ("previous_states", torch.ones(2, 3, 5)),
        ("previous_states", torch.ones(2, 3, 4, dtype=torch.int64)),
        ("previous_states", torch.full((2, 3, 4), torch.nan)),
    ],
)
def test_rejects_malformed_auxiliary_input(argument: str, replacement: object) -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    inputs: dict[str, Any] = {
        "module_inputs": prompt_context,
        "workspace_reads": workspace_reads,
        "goal_state": goal_state,
        "previous_states": None,
    }
    inputs[argument] = replacement

    with pytest.raises(CognitiveModuleError):
        model(**inputs)


@pytest.mark.parametrize("argument", ["workspace_reads", "goal_state", "previous_states"])
def test_rejects_input_dtype_mismatch(argument: str) -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    prompt_context, workspace_reads, goal_state = module_inputs()
    inputs = {
        "module_inputs": prompt_context,
        "workspace_reads": workspace_reads,
        "goal_state": goal_state,
        "previous_states": torch.zeros_like(prompt_context),
    }
    inputs[argument] = inputs[argument].to(dtype=torch.float64)

    with pytest.raises(CognitiveModuleError, match="share one dtype and device"):
        model(**inputs)


def test_rejects_parameter_dtype_mismatch() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3).to(dtype=torch.float64)
    prompt_context, workspace_reads, goal_state = module_inputs()

    with pytest.raises(CognitiveModuleError, match="parameters must explicitly match"):
        model(prompt_context, workspace_reads, goal_state)


def test_rejects_parameter_device_mismatch() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3).to(device=torch.device("meta"))
    prompt_context, workspace_reads, goal_state = module_inputs()

    with pytest.raises(CognitiveModuleError, match="parameters must explicitly match"):
        model(prompt_context, workspace_reads, goal_state)


def test_rejects_nonfinite_parameter() -> None:
    model = CognitiveModule(state_dim=4, num_modules=3)
    with torch.no_grad():
        projection = cast(nn.Linear, cast(Any, model.module_cells[2]).output_projection)
        projection.weight[0, 0] = torch.nan
    prompt_context, workspace_reads, goal_state = module_inputs()

    with pytest.raises(CognitiveModuleError, match=r"parameter .* contains NaN or Inf"):
        model(prompt_context, workspace_reads, goal_state)
