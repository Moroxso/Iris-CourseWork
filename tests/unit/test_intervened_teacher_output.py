"""Strict distinct detached NLL result and complete target identity."""

from dataclasses import replace
from typing import Any

import pytest
import torch

from iris.training.intervention import IrisIntervenedTeacherOutputV1, teacher_target_sha256_v1
from iris.workspace.state import WorkspaceState


def output() -> IrisIntervenedTeacherOutputV1:
    return IrisIntervenedTeacherOutputV1(
        loss=torch.tensor(1.0),
        supervised_token_count=2,
        workspace_trace=(
            WorkspaceState.empty(
                batch_size=1,
                num_slots=2,
                state_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        ),
        prompt_sha256="a" * 64,
        target_sha256="b" * 64,
        intervention_plan_sha256="c" * 64,
        teacher_graph_sha256="d" * 64,
        sidecar_state_sha256="e" * 64,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "baseline"),
        ("semantics", "unknown"),
        ("supervised_token_count", True),
        ("supervised_token_count", 0),
        ("supervised_token_count", 1.0),
        ("loss", torch.tensor(-1.0)),
        ("loss", torch.tensor(float("nan"))),
        ("loss", torch.tensor(float("inf"))),
        ("loss", torch.tensor([1.0])),
        ("loss", torch.tensor(1)),
        ("loss", torch.tensor(1.0, dtype=torch.float16)),
        ("loss", torch.tensor(1.0, requires_grad=True)),
        ("workspace_trace", ()),
        ("workspace_trace", []),
        ("workspace_trace", (None,)),
        ("prompt_sha256", "A" * 64),
        ("target_sha256", "b" * 63),
        ("intervention_plan_sha256", "z" * 64),
        ("teacher_graph_sha256", None),
        ("sidecar_state_sha256", 123),
    ],
)
def test_rejects_mislabeled_or_nondetached_measurements(field: str, value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(output(), **{field: value})


def test_rejects_inference_tensor_and_accepts_float64() -> None:
    baseline = output()
    with torch.inference_mode():
        loss = torch.tensor(1.0)
    with pytest.raises(ValueError):
        replace(baseline, loss=loss)
    assert (
        replace(baseline, loss=torch.tensor(2.0, dtype=torch.float64)).loss.dtype == torch.float64
    )


def test_target_digest_binds_padding_mask_and_tokens_without_mutation() -> None:
    tokens = torch.tensor([[3, 1, 0]])
    mask = torch.tensor([[True, True, False]])
    original = teacher_target_sha256_v1(tokens, mask)
    assert original == teacher_target_sha256_v1(tokens.clone(), mask.clone())
    assert original != teacher_target_sha256_v1(torch.tensor([[3, 1, 2]]), mask)
    assert original != teacher_target_sha256_v1(tokens, torch.ones_like(mask))
    assert torch.equal(tokens, torch.tensor([[3, 1, 0]]))
