"""Actual tiny-Qwen mode-bound stream, cursor, stochastic runtime and ownership."""

import json
import random
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any, Literal
from unittest.mock import PropertyMock, patch

import pytest
import torch

import iris.application as application
from iris.ablations import CognitiveAblationPolicyV1
from iris.application.scientific_ablation_optimization import IrisScientificAblationOptimizerV1
from iris.application.scientific_ablation_run import (
    IrisScientificAblationRunError,
    IrisScientificAblationRunV1,
)
from iris.application.scientific_ablation_teacher import IrisScientificAblationTeacherV1
from iris.application.scientific_checkpoint import (
    IrisScientificCheckpointError,
    build_scientific_checkpoint_v1,
)
from iris.application.scientific_delivery import (
    IrisScientificDeliveryError,
    build_scientific_delivery_v1,
)
from iris.application.scientific_optimization import (
    IrisScientificOptimizerPolicyV1,
    _same_runtime,
)
from iris.application.scientific_run import IrisScientificRunError, IrisScientificRunV1
from iris.data.generator import generate_synthetic_split
from iris.data.schemas import IrisExample, SyntheticSplitPlan
from iris.utils.hashing import canonical_json_bytes, domain_sha256, module_state_sha256
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_cognitive_ablation_teacher import _policy as _ablation
from tests.integration.test_scientific_checkpoint import _run as _baseline_run
from tests.integration.test_scientific_checkpoint import _stochastic as _baseline_stochastic
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _build, _rows

pytestmark = pytest.mark.integration


def _run(
    routing: Literal["learned", "random", "uniform"] = "learned",
    *,
    goal: str = "enabled",
    broadcast: str = "enabled",
    routing_seed: int = 42,
    run_seed: int = 901,
    model_seed: int = 1907,
    cycles: int = 3,
    examples: tuple[IrisExample, ...] | None = None,
    policy: IrisScientificOptimizerPolicyV1 | None = None,
) -> IrisScientificAblationRunV1:
    caller = capture_runtime_state(torch.device("cpu"))
    try:
        base, _, _ = _build(seed=model_seed, cycles=cycles)
        model = IrisScientificAblationTeacherV1(
            base=base,
            ablation=CognitiveAblationPolicyV1.model_validate(
                {
                    **_ablation(routing, goal=goal, broadcast=broadcast).model_dump(mode="python"),
                    "routing_seed": routing_seed if routing == "random" else None,
                }
            ),
        )
        return IrisScientificAblationRunV1(
            optimizer=IrisScientificAblationOptimizerV1(
                model=model,
                policy=_policy(minimum_target_tokens=70) if policy is None else policy,
            ),
            examples=_rows()[:5] if examples is None else examples,
            terminal_token_id=1,
            seed=run_seed,
        )
    finally:
        restore_runtime_state(caller)


@contextmanager
def _stochastic(run: IrisScientificAblationRunV1) -> Iterator[None]:
    original = run._optimizer._model.training_forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        factor = 0.9 + 0.1 * float(torch.rand(())) + 0.01 * random.gauss(0, 1)
        return replace(
            output,
            losses=replace(output.losses, language_model=output.losses.language_model * factor),
        )

    with patch.object(run._optimizer._model, "training_forward", side_effect=forward):
        yield


def _assert_equal(left: IrisScientificAblationRunV1, right: IrisScientificAblationRunV1) -> None:
    assert left.configuration_json == right.configuration_json
    assert left.configuration_sha256 == right.configuration_sha256
    assert left._cursor() == right._cursor()
    assert _same_runtime(left._runtime, right._runtime)
    a, b = left._optimizer, right._optimizer
    assert a._parameter_steps == b._parameter_steps
    assert module_state_sha256((("scientific_ablation_teacher", a._model),)) == module_state_sha256(
        (("scientific_ablation_teacher", b._model),)
    )
    assert a._topology.names == b._topology.names
    for p, q in zip(a._topology.parameters, b._topology.parameters, strict=True):
        assert torch.equal(p, q) and p.grad is None and q.grad is None
        assert (p in a._optimizer.state) == (q in b._optimizer.state)
        if p in a._optimizer.state:
            for name in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(a._optimizer.state[p][name], b._optimizer.state[q][name])


@pytest.mark.parametrize("routing", ["learned", "random", "uniform"])
@pytest.mark.parametrize("goal", ["enabled", "disabled"])
@pytest.mark.parametrize("broadcast", ["enabled", "disabled"])
def test_all_modes_have_bitwise_reproducible_private_stochastic_streams(
    routing: Literal["learned", "random", "uniform"], goal: str, broadcast: str
) -> None:
    left = _run(routing, goal=goal, broadcast=broadcast)
    right = _run(routing, goal=goal, broadcast=broadcast)
    ambient = capture_runtime_state(torch.device("cpu"))
    initial = left.history_sha256
    for _ in range(3):
        with _stochastic(left):
            expected = left.step()
        assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
        with _stochastic(right):
            assert right.step() == expected
        assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
        _assert_equal(left, right)
    assert left.completed_steps == 3 and left.examples_seen > len(left._rows)
    assert left.history_sha256 != initial and not left.poisoned
    assert all(p.grad is None for p in left._backbone._model.parameters())


@pytest.mark.parametrize(
    "kind", ["mode", "routing_seed", "run_seed", "row_order", "numeric_policy"]
)
def test_configurations_bind_independent_seeds_modes_corpus_and_numeric_policy(kind: str) -> None:
    left = _run("random")
    right = _run(
        "uniform" if kind == "mode" else "random",
        routing_seed=1337 if kind == "routing_seed" else 42,
        run_seed=1337 if kind == "run_seed" else 901,
        examples=tuple(reversed(_rows()[:5])) if kind == "row_order" else None,
        policy=_policy(minimum_target_tokens=70, max_gradient_norm=0.001)
        if kind == "numeric_policy"
        else None,
    )
    assert left.configuration_sha256 != right.configuration_sha256
    assert left.history_sha256 != right.history_sha256
    a, b = left._optimizer._model, right._optimizer._model
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert name == other and torch.equal(p, q)  # Neither seed replaces model initialization.
    assert left._backbone_identity == right._backbone_identity
    configuration = json.loads(right.configuration_json)
    assert configuration["seed"] == (1337 if kind == "run_seed" else 901)
    assert configuration["ablation"]["routing_seed"] == (
        None if kind == "mode" else 1337 if kind == "routing_seed" else 42
    )
    assert right.configuration_sha256 == domain_sha256(
        "iris/scientific-ablation-run-configuration/v1", right.configuration_json.encode()
    )
    assert right.history_sha256 == domain_sha256(
        "iris/scientific-ablation-run-history/v1", right.configuration_sha256.encode("ascii")
    )


def test_normal_control_matches_baseline_numeric_stream_but_not_its_domains() -> None:
    baseline = _baseline_run(minimum=70)
    control = _run()
    assert control.configuration_sha256 != baseline.configuration_sha256
    assert control.history_sha256 != baseline._history_sha256
    for _ in range(3):
        with _baseline_stochastic(baseline):
            expected = baseline.step()
        with _stochastic(control):
            actual = control.step()
        for name, value in asdict(expected).items():
            if name not in ("configuration_sha256", "window_sha256"):
                assert getattr(actual, name) == value
        for p, q in zip(
            baseline._optimizer._topology.parameters,
            control._optimizer._topology.parameters,
            strict=True,
        ):
            assert torch.equal(p, q)
        assert _same_runtime(baseline._runtime, control._runtime)
    assert baseline.examples_seen == control.examples_seen
    assert baseline.target_tokens_seen == control.target_tokens_seen


@pytest.mark.parametrize("seed", [None, True, 0.5, -1, 2**63, "42"])
def test_run_seed_must_be_an_explicit_bounded_integer(seed: Any) -> None:
    with pytest.raises(IrisScientificAblationRunError):
        _run(run_seed=seed)


@pytest.mark.parametrize(
    "kind", ["empty", "list", "duplicate", "held_out", "wrong_type", "invalid"]
)
def test_corpus_is_complete_unique_revalidated_train_rows(kind: str) -> None:
    rows: Any = _rows()[:5]
    if kind == "empty":
        rows = ()
    elif kind == "list":
        rows = list(rows)
    elif kind == "duplicate":
        rows = (rows[0], rows[0])
    elif kind == "held_out":
        rows = tuple(
            generate_synthetic_split(SyntheticSplitPlan(split="test", first_seed=0, count=15))
        )
    elif kind == "wrong_type":
        rows = (object(),)
    else:
        rows = (rows[0].model_copy(update={"prompt": "PRIVATE\0INVALID"}),)
    with pytest.raises(IrisScientificAblationRunError) as error:
        _run(examples=rows)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("terminal", [None, True, -1, 1.0, 999999])
def test_declared_exact_terminal_is_required_and_failed_attachment_releases_only_its_token(
    terminal: Any,
) -> None:
    run = _run()
    owner = run._optimizer
    token = owner._attached_run_token
    # Invalid second attachment must never clear the first run's live token.
    with pytest.raises(IrisScientificAblationRunError):
        IrisScientificAblationRunV1(
            optimizer=owner, examples=_rows()[:5], terminal_token_id=terminal, seed=42
        )
    assert owner._attached_run_token is token
    fresh, _, _ = _build()
    other = IrisScientificAblationOptimizerV1(
        model=IrisScientificAblationTeacherV1(base=fresh, ablation=_ablation()), policy=_policy()
    )
    with pytest.raises(IrisScientificAblationRunError):
        IrisScientificAblationRunV1(
            optimizer=other, examples=_rows()[:5], terminal_token_id=terminal, seed=42
        )
    assert other._attached_run_token is None


def test_only_exact_fresh_optimizer_can_attach_and_legacy_io_rejects_new_run() -> None:
    run = _run()
    assert application.IrisScientificAblationRunV1 is IrisScientificAblationRunV1
    assert application.IrisScientificAblationRunError is IrisScientificAblationRunError
    with pytest.raises(IrisScientificRunError):
        IrisScientificRunV1(
            optimizer=run._optimizer,  # type: ignore[arg-type]
            examples=_rows(),
            terminal_token_id=1,
            seed=42,
        )
    baseline = _baseline_run()
    with pytest.raises(IrisScientificAblationRunError):
        IrisScientificAblationRunV1(
            optimizer=baseline._optimizer,  # type: ignore[arg-type]
            examples=_rows(),
            terminal_token_id=1,
            seed=42,
        )
    with pytest.raises(IrisScientificCheckpointError):
        build_scientific_checkpoint_v1(run)  # type: ignore[arg-type]
    with pytest.raises(IrisScientificDeliveryError):
        build_scientific_delivery_v1(run)  # type: ignore[arg-type]
    run._optimizer.step(_rows()[:2], terminal_token_id=1)
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run.poisoned and run.completed_steps == 0


def test_external_input_facts_do_not_alias_owned_rows_and_selected_private_mutation_rejects() -> (
    None
):
    rows = _rows()[:5]
    run = _run(examples=rows)
    original = canonical_json_bytes(run._rows[0].model_dump(mode="json"))
    rows[0].hidden_facts["active_constraint"] = "PRIVATE input mutation"
    assert canonical_json_bytes(run._rows[0].model_dump(mode="json")) == original
    run.step()
    row = run._rows[run.examples_seen % len(run._rows)]
    row.hidden_facts["active_constraint"] = "PRIVATE owned mutation"
    cursor = run._cursor()
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run._cursor() == cursor and run.poisoned


@pytest.mark.parametrize(
    "kind",
    [
        "examples",
        "tokens",
        "steps",
        "history",
        "rng",
        "python_rng",
        "domain",
        "configuration",
        "attachment",
    ],
)
def test_cursor_runtime_configuration_and_attachment_drift_reject_before_forward(kind: str) -> None:
    run = _run()
    if kind == "examples":
        run._examples_seen += 1
    elif kind == "tokens":
        run._target_tokens_seen += 1
    elif kind == "steps":
        run._completed_steps += 1
    elif kind == "history":
        run._history_sha256 = "0" * 64
    elif kind == "rng":
        run._runtime.cpu_rng_state[0] ^= 1
    elif kind == "python_rng":
        run._runtime = replace(run._runtime, python_rng_state=random.Random(42).getstate())
    elif kind == "domain":
        run._history_domain = "iris/scientific-run-history/v1"
    elif kind == "configuration":
        run._configuration_json = "{}"
    else:
        run._optimizer._attached_run_token = object()
    ambient = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run._optimizer.completed_steps == 0 and run.poisoned
    assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "field",
    [
        "step_index",
        "example_count",
        "supervised_token_count",
        "purpose",
        "configuration_sha256",
        "model_configuration_sha256",
        "ablation_policy_sha256",
        "scientific_state_before_sha256",
        "scientific_state_after_sha256",
        "window_sha256",
    ],
)
def test_wrong_execution_provenance_after_real_update_poisons_without_cursor_commit(
    field: str,
) -> None:
    run = _run()
    owner = run._optimizer
    original = owner.step
    before = run._cursor()
    runtime = run._clone_runtime(run._runtime)
    caller = capture_runtime_state(torch.device("cpu"))

    def wrong(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        value: Any = getattr(result, field)
        changes: dict[str, Any] = {
            field: value + 1
            if type(value) is int
            else "scientific"
            if field == "purpose"
            else "0" * 64
        }
        return replace(result, **changes)

    with (
        patch.object(owner, "step", side_effect=wrong),
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    assert owner.completed_steps == 1 and run._cursor() == before and run.poisoned
    assert _same_runtime(runtime, run._runtime)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_same_size_reordered_execution_cannot_masquerade_as_selected_window() -> None:
    run = _run()
    original = run._optimizer.step

    def wrong(window: tuple[IrisExample, ...], **kwargs: Any) -> Any:
        assert len(window) > 1
        return original(tuple(reversed(window)), **kwargs)

    with (
        patch.object(run._optimizer, "step", side_effect=wrong),
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    assert run._optimizer.completed_steps == 1 and run.completed_steps == 0 and run.poisoned


@pytest.mark.parametrize("kind", ["minimum", "examples", "maximum"])
def test_unmet_window_budget_has_no_forward_update_cursor_or_private_runtime(kind: str) -> None:
    policy = _policy(
        minimum_target_tokens=10000 if kind == "minimum" else 70 if kind == "examples" else 1,
        maximum_examples=1 if kind == "examples" else 4096,
        maximum_target_tokens=1 if kind == "maximum" else 65536,
    )
    run = _run(policy=policy)
    before, runtime = run._cursor(), run._clone_runtime(run._runtime)
    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run._cursor() == before and not run.poisoned
    assert _same_runtime(runtime, run._runtime)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_replay_failure_has_exact_rng_rollback_and_safe_retry() -> None:
    run = _run("random")
    model = run._optimizer._model
    original = model.training_forward
    before, runtime = run._cursor(), run._clone_runtime(run._runtime)
    caller = capture_runtime_state(torch.device("cpu"))
    calls = 0

    def drift(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        output = original(*args, **kwargs)
        if calls > 2:
            return replace(
                output,
                losses=replace(output.losses, language_model=output.losses.language_model + 1),
            )
        return output

    with (
        patch.object(model, "training_forward", side_effect=drift),
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    assert run._cursor() == before and not run.poisoned
    assert _same_runtime(runtime, run._runtime)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert run.step().step_index == 1


def test_reentry_preserves_outer_provenance_and_commits_once() -> None:
    run = _run("random", policy=_policy())
    original = run._optimizer._model.training_forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(IrisScientificAblationRunError):
            run.step()
        return original(*args, **kwargs)

    with patch.object(run._optimizer._model, "training_forward", side_effect=forward):
        assert run.step().step_index == 1
    assert run.completed_steps == 1 and not run.poisoned


def test_actual_frozen_identity_is_read_once_on_attachment_not_each_step() -> None:
    base, _, _ = _build()
    owner = IrisScientificAblationOptimizerV1(
        model=IrisScientificAblationTeacherV1(base=base, ablation=_ablation("random")),
        policy=_policy(),
    )
    backbone = owner._model.teacher.backbone
    identity = backbone.checkpoint_runtime_sha256
    with patch.object(
        type(backbone),
        "checkpoint_runtime_sha256",
        new_callable=PropertyMock,
        return_value=identity,
    ) as read:
        run = IrisScientificAblationRunV1(
            optimizer=owner, examples=_rows()[:5], terminal_token_id=1, seed=42
        )
        assert read.call_count == 1 and run._backbone_identity == identity
        run.step()
        run.step()
        assert read.call_count == 1


def test_software_drift_rejects_before_forward_and_unexecuted_policy_restoration_allows_retry() -> (
    None
):
    run = _run()
    with (
        patch(
            "iris.application.scientific_run.scientific_software_v1",
            return_value={**run._software, "torch_threads": "999"},
        ),
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert not run.poisoned and run.completed_steps == 0
    assert run.step().step_index == 1


@pytest.mark.parametrize("kind", ["optimizer", "commit", "caller_restore"])
def test_uncertain_update_commit_or_restoration_failure_poisons_and_redacts(kind: str) -> None:
    run = _run(policy=_policy())
    ambient = capture_runtime_state(torch.device("cpu"))
    if kind == "optimizer":
        context = patch.object(
            run._optimizer._optimizer, "step", side_effect=RuntimeError("PRIVATE failure")
        )
    elif kind == "commit":
        context = patch.object(run, "_commit_owner", side_effect=RuntimeError("PRIVATE failure"))
    else:
        calls = 0

        def restore(value: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("PRIVATE failure")
            restore_runtime_state(value)

        context = patch(
            "iris.application.scientific_run.restore_runtime_state", side_effect=restore
        )
    try:
        with context, pytest.raises(IrisScientificAblationRunError) as error:
            run.step()
        assert "PRIVATE" not in str(error.value) and run.poisoned
        assert run._optimizer.completed_steps == (0 if kind == "optimizer" else 1)
        with pytest.raises(IrisScientificAblationRunError):
            run.step()
    finally:
        restore_runtime_state(ambient)
    assert all(p.grad is None for p in run._optimizer._model.parameters())


def test_source_sized_ablation_run_crosses_epoch_without_tail_drop_or_duplicate_rows(
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    rows = tuple(
        generate_synthetic_split(
            SyntheticSplitPlan(split="train", first_seed=0, count=600),
            template_id="iris-synthetic-01-v2",
        )
    )
    run = _run(
        "random",
        goal="disabled",
        broadcast="disabled",
        cycles=1,
        examples=rows,
        policy=IrisScientificOptimizerPolicyV1(),
    )
    ambient = capture_runtime_state(torch.device("cpu"))
    previous = run.history_sha256
    outputs = []
    for _ in range(2):
        start = run.examples_seen
        with patch.object(run._optimizer, "step", wraps=run._optimizer.step) as step:
            output = run.step()
        selected = step.call_args.args[0]
        assert len(selected) == len({row.id for row in selected}) == output.example_count
        assert selected == tuple(
            rows[(start + offset) % len(rows)] for offset in range(len(selected))
        )
        if start:
            assert rows[-1] in selected and rows[0] in selected
        assert output.purpose == "scientific" and output.supervised_token_count >= 16384
        assert run.history_sha256 == domain_sha256(
            "iris/scientific-ablation-run-history/v1",
            canonical_json_bytes(
                {
                    "previous": previous,
                    "result": asdict(output),
                    "examples_seen": run.examples_seen,
                    "target_tokens_seen": run.target_tokens_seen,
                }
            ),
        )
        previous = run.history_sha256
        outputs.append(output)
        assert _same_runtime(ambient, capture_runtime_state(torch.device("cpu")))
    assert run.examples_seen == 904 and run.target_tokens_seen == 32808
    assert run._optimizer._policy == IrisScientificOptimizerPolicyV1() and not run.poisoned
    for name, value in (
        ("cycles", 1),
        ("first_examples", outputs[0].example_count),
        ("first_target_tokens", outputs[0].supervised_token_count),
        ("second_examples", outputs[1].example_count),
        ("second_target_tokens", outputs[1].supervised_token_count),
        ("examples_seen", run.examples_seen),
        ("target_tokens_seen", run.target_tokens_seen),
        ("history_sha256", run.history_sha256),
    ):
        record_testsuite_property("scientific_ablation_run_" + name, value)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_constructor_signal_releases_only_its_attachment_and_preserves_signal(
    signal: type[BaseException],
) -> None:
    base, _, _ = _build()
    owner = IrisScientificAblationOptimizerV1(
        model=IrisScientificAblationTeacherV1(base=base, ablation=_ablation()), policy=_policy()
    )
    with (
        patch.object(IrisScientificAblationRunV1, "_validate_boundary", side_effect=signal()),
        pytest.raises(signal),
    ):
        IrisScientificAblationRunV1(optimizer=owner, examples=_rows(), terminal_token_id=1, seed=42)
    assert owner._attached_run_token is None and not owner.poisoned
    run = IrisScientificAblationRunV1(
        optimizer=owner, examples=_rows(), terminal_token_id=1, seed=42
    )
    assert run.step().step_index == 1


def test_failed_initial_boundary_releases_attachment_without_erasing_existing_gradient() -> None:
    base, _, _ = _build()
    owner = IrisScientificAblationOptimizerV1(
        model=IrisScientificAblationTeacherV1(base=base, ablation=_ablation()), policy=_policy()
    )
    parameter = owner._topology.parameters[0]
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(IrisScientificAblationRunError):
        IrisScientificAblationRunV1(optimizer=owner, examples=_rows(), terminal_token_id=1, seed=42)
    assert owner._attached_run_token is None and parameter.grad is not None
    parameter.grad = None  # The test, not attachment failure, cleans its own fixture.
    assert (
        IrisScientificAblationRunV1(
            optimizer=owner, examples=_rows(), terminal_token_id=1, seed=42
        ).completed_steps
        == 0
    )


@pytest.mark.parametrize("kind", ["evaluation", "no_grad", "autocast"])
def test_explicit_train_autograd_fp32_boundary_rejects_before_forward(kind: str) -> None:
    from contextlib import nullcontext

    run = _run()
    if kind == "evaluation":
        run._optimizer._model.eval()
    context = (
        torch.no_grad()
        if kind == "no_grad"
        else torch.autocast("cpu", dtype=torch.bfloat16)
        if kind == "autocast"
        else nullcontext()
    )
    with (
        context,
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run.completed_steps == 0 and run._optimizer.completed_steps == 0


def test_changed_frozen_revision_rejects_before_forward() -> None:
    run = _run()
    with torch.no_grad():
        next(run._backbone._model.parameters()).add_(0.01)
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run.completed_steps == 0 and run._optimizer.completed_steps == 0


def test_run_installs_its_policy_and_restores_different_caller_runtime_exactly() -> None:
    run = _run(policy=_policy())
    ambient = capture_runtime_state(torch.device("cpu"))
    try:
        torch.use_deterministic_algorithms(not run._runtime.deterministic_algorithms)
        caller = capture_runtime_state(torch.device("cpu"))
        original = run._optimizer._model.training_forward

        def forward(*args: Any, **kwargs: Any) -> Any:
            assert (
                torch.are_deterministic_algorithms_enabled()
                == run._runtime.deterministic_algorithms
            )
            return original(*args, **kwargs)

        with patch.object(run._optimizer._model, "training_forward", side_effect=forward):
            run.step()
        assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    finally:
        restore_runtime_state(ambient)


def test_schedule_exhaustion_does_not_forward_or_extend_history() -> None:
    run = _run(policy=_policy(max_optimizer_steps=2))
    run.step()
    run.step()
    cursor = run._cursor()
    with (
        patch.object(run._optimizer._model, "training_forward") as forward,
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    forward.assert_not_called()
    assert run._cursor() == cursor and not run.poisoned


@pytest.mark.parametrize("kind", ["cursor", "private_rng"])
def test_child_cannot_rewrite_parent_owned_state_and_have_it_silently_rebound(kind: str) -> None:
    run = _run(policy=_policy())
    original = run._optimizer._model.training_forward
    caller = capture_runtime_state(torch.device("cpu"))

    def forward(*args: Any, **kwargs: Any) -> Any:
        if kind == "cursor":
            run._history_sha256 = "0" * 64
        else:
            run._runtime.cpu_rng_state[0] = 0
        return original(*args, **kwargs)

    with (
        patch.object(run._optimizer._model, "training_forward", side_effect=forward),
        pytest.raises(IrisScientificAblationRunError),
    ):
        run.step()
    assert run._optimizer.completed_steps == 1 and run.completed_steps == 0 and run.poisoned
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_nonfresh_optimizer_rejects_without_claiming_attachment() -> None:
    base, _, _ = _build()
    owner = IrisScientificAblationOptimizerV1(
        model=IrisScientificAblationTeacherV1(base=base, ablation=_ablation()), policy=_policy()
    )
    owner.step(_rows()[:1], terminal_token_id=1)
    with pytest.raises(IrisScientificAblationRunError):
        IrisScientificAblationRunV1(optimizer=owner, examples=_rows(), terminal_token_id=1, seed=42)
    assert owner._attached_run_token is None and owner.completed_steps == 1
