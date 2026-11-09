"""Source-sized token windows through the actual run owner and nine-file resume."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from iris.application.scientific_checkpoint import build_scientific_checkpoint_v1
from iris.application.scientific_delivery import (
    build_scientific_delivery_v1,
    read_scientific_delivery_v1,
    restore_scientific_delivery_v1,
    write_scientific_delivery_v1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizerPolicyV1,
    IrisScientificOptimizerV1,
    _same_runtime,
)
from iris.application.scientific_run import IrisScientificRunV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.utils.seeds import capture_runtime_state
from tests.integration.test_scientific_checkpoint import _stochastic
from tests.integration.test_scientific_teacher import _build

pytestmark = pytest.mark.integration


def _scale_run(rows: tuple[IrisExample, ...]) -> IrisScientificRunV1:
    model, _, _ = _build(cycles=1)
    return IrisScientificRunV1(
        optimizer=IrisScientificOptimizerV1(model=model, policy=IrisScientificOptimizerPolicyV1()),
        examples=rows,
        terminal_token_id=1,
        seed=901,
    )


def test_real_default_run_windows_cross_epoch_and_resume_exactly(
    tmp_path: Path, record_testsuite_property: Callable[[str, object], None]
) -> None:
    ambient = capture_runtime_state(torch.device("cpu"))
    rows = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=600),
            template_id="iris-synthetic-01-v2",
        )
    )
    run = _scale_run(rows)
    policy = run._optimizer._policy
    assert policy.purpose == "scientific" and policy.minimum_target_tokens == 16384
    assert policy == IrisScientificOptimizerPolicyV1()  # No smoke window override.
    counts = [
        int(
            run._backbone.tokenize_teacher_target(
                row.expected_response, terminal_token_id=1
            ).attention_mask.sum()
        )
        for row in rows
    ]

    def expected_window(start: int) -> tuple[tuple[str, ...], int]:
        identities: list[str] = []
        total = 0
        while total < policy.minimum_target_tokens:
            offset = (start + len(identities)) % len(rows)
            identities.append(rows[offset].id)
            total += counts[offset]
        return tuple(identities), total

    first_ids, first_tokens = expected_window(0)
    assert len(first_ids) == 452 and first_tokens == 16420
    with _stochastic(run), patch.object(run._optimizer, "step", wraps=run._optimizer.step) as step:
        first = run.step()
    assert tuple(row.id for row in step.call_args.args[0]) == first_ids
    assert first.example_count == len(first_ids) and first.supervised_token_count == first_tokens
    assert run.examples_seen == len(first_ids) and run.target_tokens_seen == first_tokens
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    checkpoint = build_scientific_checkpoint_v1(run)
    delivery = build_scientific_delivery_v1(run)
    path = write_scientific_delivery_v1(delivery, root=tmp_path)
    resumed = _scale_run(rows)
    with patch("torch.load", side_effect=AssertionError("incoming pickle execution forbidden")):
        restore_scientific_delivery_v1(read_scientific_delivery_v1(path), run=resumed)
    assert build_scientific_checkpoint_v1(resumed) == checkpoint
    second_ids, second_tokens = expected_window(len(first_ids))
    assert len(first_ids) + len(second_ids) > len(rows)
    assert len(set(second_ids)) == len(second_ids)  # No duplicate inside a window.
    assert second_ids[0] == rows[len(first_ids)].id
    assert rows[-1].id in second_ids and rows[0].id in second_ids
    with _stochastic(run), patch.object(run._optimizer, "step", wraps=run._optimizer.step) as step:
        expected = run.step()
    assert tuple(row.id for row in step.call_args.args[0]) == second_ids
    with (
        _stochastic(resumed),
        patch.object(resumed._optimizer, "step", wraps=resumed._optimizer.step) as step,
    ):
        actual = resumed.step()
    assert tuple(row.id for row in step.call_args.args[0]) == second_ids
    assert actual == expected
    assert actual.supervised_token_count == second_tokens >= 16384
    assert resumed.examples_seen == len(first_ids) + len(second_ids)
    assert resumed.target_tokens_seen == first_tokens + second_tokens
    assert build_scientific_checkpoint_v1(resumed) == build_scientific_checkpoint_v1(run)
    assert build_scientific_delivery_v1(resumed) == build_scientific_delivery_v1(run)
    assert _same_runtime(capture_runtime_state(torch.device("cpu")), ambient)
    assert all(p.grad is None for p in resumed._optimizer._model.parameters())
    assert all(
        not p.requires_grad and p.grad is None for p in resumed._backbone._model.parameters()
    )
    record_testsuite_property("run_scale_cognitive_cycles", 1)
    record_testsuite_property("run_scale_first_examples", len(first_ids))
    record_testsuite_property("run_scale_first_target_tokens", first_tokens)
    record_testsuite_property("run_scale_second_examples", len(second_ids))
    record_testsuite_property("run_scale_second_target_tokens", second_tokens)
    record_testsuite_property("run_scale_examples_seen", resumed.examples_seen)
    record_testsuite_property("run_scale_target_tokens_seen", resumed.target_tokens_seen)
    record_testsuite_property(
        "run_scale_delivery_bytes", sum(len(data) for _, data in delivery.files)
    )
