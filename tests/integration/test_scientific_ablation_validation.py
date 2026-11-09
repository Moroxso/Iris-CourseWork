"""Actual tiny-Qwen mode-bound validation, NLL, non-mutation and failure ownership."""

import random
from dataclasses import asdict, replace
from math import fsum
from typing import Any
from unittest.mock import patch

import pytest
import torch

import iris.application as application
from iris.application import scientific_ablation_validation as module
from iris.application.scientific_ablation_checkpoint import (
    IrisScientificAblationCheckpointError,
)
from iris.application.scientific_ablation_checkpoint import (
    build_scientific_ablation_checkpoint_v1 as build,
)
from iris.application.scientific_optimization import _same_runtime
from iris.application.scientific_run import IrisScientificRunError
from iris.utils.hashing import canonical_json_bytes
from iris.utils.seeds import capture_runtime_state, restore_runtime_state
from tests.integration.test_scientific_ablation_run import _assert_equal, _run, _stochastic
from tests.integration.test_scientific_checkpoint import _run as _old_run
from tests.integration.test_scientific_lifecycle import _validation
from tests.integration.test_scientific_optimization import _policy
from tests.integration.test_scientific_teacher import _rows

pytestmark = pytest.mark.integration

evaluate = module.evaluate_scientific_ablation_validation_v1
own = module.own_scientific_ablation_validation_rows_v1
Error = module.IrisScientificAblationValidationError


def _short(routing: Any = "learned", *, goal: str = "enabled", broadcast: str = "enabled") -> Any:
    return _run(
        routing, goal=goal, broadcast=broadcast, cycles=3, policy=_policy(minimum_target_tokens=1)
    )


@pytest.mark.parametrize("routing", ("learned", "random", "uniform"))
@pytest.mark.parametrize("goal", ("enabled", "disabled"))
@pytest.mark.parametrize("broadcast", ("enabled", "disabled"))
def test_real_c3_mode_validation_token_nll_and_next_stochastic_update(
    routing: Any, goal: str, broadcast: str
) -> None:
    run, twin = (
        _short(routing, goal=goal, broadcast=broadcast),
        _short(routing, goal=goal, broadcast=broadcast),
    )
    for current in (run, twin):
        with _stochastic(current):
            current.step()
    assert run._runtime.python_rng_state[2] is not None
    before = build(run)
    caller = capture_runtime_state(torch.device("cpu"))
    report = evaluate(run, _validation())
    assert type(report) is module.IrisScientificAblationValidationV1
    assert report.at_optimizer_step == 1 and report.example_count == 4
    assert report.run_configuration_sha256 == run.configuration_sha256
    assert report.model_configuration_sha256 == run._optimizer._model.configuration_sha256
    assert report.ablation_policy_sha256 == run._ablation_identity
    assert report.scientific_state_sha256 == before.manifest.payload.scientific_state_sha256
    assert build(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    model = run._optimizer._model
    try:
        restore_runtime_state(run._runtime)
        model.eval()
        outputs = [model.evaluation_forward(row, terminal_token_id=1) for row in _validation()]
        sums = [float(output.losses.language_model) for output in outputs]
        counts = [output.supervised_token_count for output in outputs]
        assert len(set(counts)) > 1
        assert report.target_token_count == sum(counts)
        assert report.target_nll_sum == fsum(sums)
        assert report.target_nll_per_token == fsum(sums) / sum(counts)
        assert report.target_nll_per_token != pytest.approx(
            fsum(value / count for value, count in zip(sums, counts, strict=True)) / len(counts),
            abs=1e-8,
        )
    finally:
        model.train(True)
        restore_runtime_state(caller)
    for current in (run, twin):
        with _stochastic(current):
            result = current.step()
        if current is run:
            actual = result
        else:
            assert asdict(actual) == asdict(result)
    _assert_equal(run, twin)
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert all(child.training for child in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_owned_validation_rows_are_defensive_ordered_and_in_distinct_domain() -> None:
    rows = _validation()
    owned, digest = own(rows, training_rows=_rows()[:5])
    assert len(owned) == 4 and owned is not rows
    reversed_rows, reversed_digest = own(tuple(reversed(rows)), training_rows=_rows()[:5])
    assert reversed_digest != digest and reversed_rows[0].id == owned[-1].id
    before = canonical_json_bytes(owned[0].model_dump(mode="json"))
    rows[0].hidden_facts["result_type"] = "mutated external source"
    assert canonical_json_bytes(owned[0].model_dump(mode="json")) == before


@pytest.mark.parametrize(
    "kind",
    ("empty", "list", "train", "test", "duplicate", "overlap", "unknown_field", "nested_unknown"),
)
def test_bad_validation_corpus_rejects_without_owned_change(kind: str) -> None:
    run = _short()
    before = build(run)
    row = _validation()[0]
    examples: Any = (row,)
    if kind == "empty":
        examples = ()
    elif kind == "list":
        examples = [row]
    elif kind == "train":
        examples = (_rows()[0],)
    elif kind == "test":
        examples = (row.model_copy(update={"split": "test", "id": f"test-{row.seed}"}),)
    elif kind == "duplicate":
        examples = (row, row)
    elif kind == "overlap":
        examples = (row.model_copy(update={"seed": 0, "id": "validation-0"}),)
    elif kind == "unknown_field":
        examples = (row.model_copy(update={"not_a_field": True}),)
    else:
        examples = (
            row.model_copy(
                update={
                    "code_cases": (
                        row.code_cases[0].model_copy(update={"not_a_field": True}),
                        *row.code_cases[1:],
                    )
                }
            ),
        )
    caller = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(Error):
        evaluate(run, examples)
    assert build(run) == before and not run.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "kind",
    (
        "mode",
        "row",
        "config",
        "state",
        "invocation",
        "bool_count",
        "wrong_count",
        "nested_graph",
        "nested_sidecar",
        "nested_policy",
        "nan",
        "grad",
    ),
)
def test_detached_output_and_exact_actual_provenance_are_required(kind: str) -> None:
    run = _short()
    before = build(run)
    original = run._optimizer._model.evaluation_forward

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        changes: dict[str, Any] = {}
        if kind == "mode":
            changes["mode"] = "training"
        elif kind == "row":
            changes["example_sha256"] = "0" * 64
        elif kind == "config":
            changes["configuration_sha256"] = "0" * 64
        elif kind == "state":
            changes["scientific_state_sha256"] = "0" * 64
        elif kind == "invocation":
            changes["invocation_sha256"] = "0" * 64
        elif kind == "bool_count":
            changes["supervised_token_count"] = True
        elif kind == "wrong_count":
            changes["supervised_token_count"] = output.supervised_token_count + 1
        elif kind in ("nested_graph", "nested_sidecar"):
            field = "teacher_graph_sha256" if kind == "nested_graph" else "sidecar_state_sha256"
            changes["teacher"] = replace(output.teacher, **{field: "0" * 64})
        elif kind == "nested_policy":
            changes["teacher"] = replace(
                output.teacher, policy=output.teacher.policy.model_copy(update={"goal": "disabled"})
            )
        else:
            loss = torch.tensor(
                float("nan") if kind == "nan" else 1.0, requires_grad=kind == "grad"
            )
            changes["losses"] = replace(output.losses, language_model=loss)
        return replace(output, **changes)

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(run._optimizer._model, "evaluation_forward", side_effect=corrupt),
        pytest.raises(Error),
    ):
        evaluate(run, _validation()[:1])
    assert not run.poisoned and not run._busy and build(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert evaluate(run, _validation()[:1]).example_count == 1


@pytest.mark.parametrize(
    "failure", (ValueError("forward"), KeyboardInterrupt("forward"), SystemExit(47))
)
def test_untouched_forward_failure_restores_modes_rng_and_is_retryable(
    failure: BaseException,
) -> None:
    run = _short()
    before = build(run)
    caller = capture_runtime_state(torch.device("cpu"))

    def fail(*args: Any, **kwargs: Any) -> Any:
        random.gauss(0, 1)
        torch.rand(())
        raise failure

    expected = Error if isinstance(failure, Exception) else type(failure)
    with (
        patch.object(run._optimizer._model, "evaluation_forward", side_effect=fail),
        pytest.raises(expected),
    ):
        evaluate(run, _validation()[:1])
    assert not run.poisoned and not run._busy and build(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert all(child.training for child in run._optimizer._model.modules())
    assert evaluate(run, _validation()[:1]).example_count == 1


@pytest.mark.parametrize(
    "failure", (ValueError("partial"), KeyboardInterrupt("partial"), SystemExit(53))
)
def test_partial_model_write_poisons_both_and_preserves_signals(failure: BaseException) -> None:
    run = _short()
    caller = capture_runtime_state(torch.device("cpu"))

    def partial(*args: Any, **kwargs: Any) -> Any:
        parameter = next(run._optimizer._model.parameters())
        with torch.no_grad():
            parameter.reshape(-1)[0].add_(0.125)
        raise failure

    expected = Error if isinstance(failure, Exception) else type(failure)
    with (
        patch.object(run._optimizer._model, "evaluation_forward", side_effect=partial),
        pytest.raises(expected),
    ):
        evaluate(run, _validation()[:1])
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    with pytest.raises(IrisScientificRunError):
        run.step()


@pytest.mark.parametrize("kind", ("moment_data", "frozen_data", "resealed_cursor"))
def test_mutation_even_without_version_bump_or_with_self_seal_poisons(kind: str) -> None:
    run = _short()
    run.step()
    original = run._optimizer._model.evaluation_forward

    def mutate(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        if kind == "moment_data":
            state = next(iter(run._optimizer._optimizer.state.values()))
            state["exp_avg"].data.reshape(-1)[0].add_(0.125)
        elif kind == "frozen_data":
            parameter = next(run._backbone._model.parameters())
            parameter.data.reshape(-1)[0].add_(0.125)
        else:
            run._examples_seen += 1
            run._commit_owner()
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(run._optimizer._model, "evaluation_forward", side_effect=mutate),
        pytest.raises(Error),
    ):
        evaluate(run, _validation()[:1])
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_preexisting_frozen_data_mutation_cannot_be_hidden_as_new_valid_state() -> None:
    run = _short()
    next(run._backbone._model.parameters()).data.reshape(-1)[0].add_(0.125)
    caller = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(Error):
        evaluate(run, _validation()[:1])
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_validation_reentry_training_and_checkpoint_are_rejected() -> None:
    run = _short()
    before = build(run)
    original = run._optimizer._model.evaluation_forward

    def nested(*args: Any, **kwargs: Any) -> Any:
        with pytest.raises(Error):
            evaluate(run, _validation()[:1])
        with pytest.raises(IrisScientificRunError):
            run.step()
        with pytest.raises(IrisScientificAblationCheckpointError):
            build(run)
        return original(*args, **kwargs)

    with patch.object(run._optimizer._model, "evaluation_forward", side_effect=nested):
        assert evaluate(run, _validation()[:1]).example_count == 1
    assert not run.poisoned and not run._busy and build(run) == before


@pytest.mark.parametrize("primary_signal", (False, True))
def test_uncertain_caller_restore_poisons_both_and_keeps_primary_signal(
    primary_signal: bool,
) -> None:
    run = _short()
    caller = capture_runtime_state(torch.device("cpu"))
    original_restore = restore_runtime_state
    calls = 0

    def uncertain(snapshot: Any) -> None:
        nonlocal calls
        calls += 1
        original_restore(snapshot)
        if calls == 2:
            raise ValueError("lost caller restoration acknowledgment")

    with patch.object(module, "restore_runtime_state", side_effect=uncertain):
        if primary_signal:
            with (
                patch.object(
                    run._optimizer._model,
                    "evaluation_forward",
                    side_effect=KeyboardInterrupt("primary"),
                ),
                pytest.raises(KeyboardInterrupt, match="primary"),
            ):
                evaluate(run, _validation()[:1])
        else:
            with pytest.raises(Error):
                evaluate(run, _validation()[:1])
    assert calls == 2 and run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_mode_restoration_failure_poisons_both_and_releases_lock() -> None:
    run = _short()
    model = run._optimizer._model
    original_train = model.train
    caller = capture_runtime_state(torch.device("cpu"))

    def train(mode: bool = True) -> Any:
        if mode:
            raise ValueError("mode cleanup")
        return original_train(mode)

    with patch.object(model, "train", side_effect=train), pytest.raises(Error):
        evaluate(run, _validation()[:1])
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_historical_exact_run_is_not_an_ablation_owner() -> None:
    old = _old_run()
    caller = capture_runtime_state(torch.device("cpu"))
    with pytest.raises(Error):
        evaluate(old, _validation()[:1])  # type: ignore[arg-type]
    assert not old.poisoned and not old._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_canonical_public_exports_and_distinct_baseline_report_domain() -> None:
    from iris.application.scientific_validation import (
        IrisScientificValidationError,
        IrisScientificValidationV1,
        own_scientific_validation_rows_v1,
    )

    for name in module.__all__:
        assert name in application.__all__
        assert getattr(application, name) is getattr(module, name)
    historical_domains = {
        "IrisScientificAblationValidationError": IrisScientificValidationError,
        "IrisScientificAblationValidationV1": IrisScientificValidationV1,
    }
    for name, historical in historical_domains.items():
        assert getattr(application, name) is not historical
    rows = _validation()
    _, ablation_digest = own(rows, training_rows=_rows()[:5])
    _, baseline_digest = own_scientific_validation_rows_v1(rows, training_rows=_rows()[:5])
    assert ablation_digest != baseline_digest


@pytest.mark.parametrize("when", ("before", "during"))
def test_owned_training_corpus_mutation_rejects_and_poisons_both(when: str) -> None:
    run = _short()
    model = run._optimizer._model
    original = model.evaluation_forward

    def mutate() -> None:
        run._rows[0].hidden_facts["result_type"] = "changed owned training facts"

    def forward(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        mutate()
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    if when == "before":
        mutate()
        with pytest.raises(Error):
            evaluate(run, _validation()[:1])
    else:
        with patch.object(model, "evaluation_forward", side_effect=forward), pytest.raises(Error):
            evaluate(run, _validation()[:1])
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("when", ("before", "during"))
@pytest.mark.parametrize("kind", ("unknown_row_field", "unknown_nested_field"))
def test_unknown_training_fields_reject_before_serialization(when: str, kind: str) -> None:
    run = _short()
    row = run._rows[0]
    original_forward = run._optimizer._model.evaluation_forward
    original_dump = type(row).model_dump
    serialized_after_mutation = False
    mutated = False

    def mutate() -> None:
        nonlocal mutated
        target = row if kind == "unknown_row_field" else row.code_cases[0]
        vars(target)["not_a_field"] = True
        mutated = True

    def dump(current: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal serialized_after_mutation
        if current is row and mutated:
            serialized_after_mutation = True
        return original_dump(current, *args, **kwargs)

    def forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        mutate()
        return output

    caller = capture_runtime_state(torch.device("cpu"))
    if when == "before":
        mutate()
    with (
        patch.object(type(row), "model_dump", new=dump),
        patch.object(run._optimizer._model, "evaluation_forward", side_effect=forward),
        pytest.raises(Error),
    ):
        evaluate(run, _validation()[:1])
    assert not serialized_after_mutation
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize("kind", ("unknown_row_field", "unknown_nested_field"))
def test_public_corpus_owner_rejects_unchecked_training_fields(kind: str) -> None:
    row = _rows()[0]
    target = row if kind == "unknown_row_field" else row.code_cases[0]
    vars(target)["not_a_field"] = True
    with pytest.raises(Error):
        own(_validation()[:1], training_rows=(row,))


def test_original_training_seed_cannot_be_laundered_before_validation() -> None:
    run = _short()
    row = run._rows[0]
    validation = row.model_copy(update={"split": "validation", "id": f"validation-{row.seed}"})
    object.__setattr__(row, "seed", 20000001)
    object.__setattr__(row, "id", "train-20000001")
    caller = capture_runtime_state(torch.device("cpu"))
    with patch.object(module, "own_scientific_ablation_validation_rows_v1", wraps=own) as owner:
        with pytest.raises(Error):
            evaluate(run, (validation,))
        owner.assert_not_called()
    assert run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


def test_training_serialization_cannot_reenter_validation_step_or_checkpoint() -> None:
    run = _short()
    before = build(run)
    row = run._rows[0]
    original_dump = type(row).model_dump
    checked = 0

    def dump(current: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal checked
        if current is row:
            checked += 1
            with pytest.raises(Error):
                evaluate(run, _validation()[:1])
            with pytest.raises(IrisScientificRunError):
                run.step()
            with pytest.raises(IrisScientificAblationCheckpointError):
                build(run)
        return original_dump(current, *args, **kwargs)

    caller = capture_runtime_state(torch.device("cpu"))
    with patch.object(type(row), "model_dump", new=dump):
        assert evaluate(run, _validation()[:1]).example_count == 1
    assert checked == 2
    assert not run.poisoned and not run._busy and build(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))


@pytest.mark.parametrize(
    "failure", (MemoryError("snapshot"), KeyboardInterrupt("snapshot"), SystemExit(61))
)
def test_untouched_initial_snapshot_failure_is_retryable(failure: BaseException) -> None:
    run = _short()
    before = build(run)
    original = module._snapshot
    calls = 0

    def snapshot(current: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            random.gauss(0, 1)
            torch.rand(())
            raise failure
        return original(current)

    caller = capture_runtime_state(torch.device("cpu"))
    expected = Error if isinstance(failure, Exception) else type(failure)
    with patch.object(module, "_snapshot", side_effect=snapshot), pytest.raises(expected):
        evaluate(run, _validation()[:1])
    assert calls == 2
    assert not run.poisoned and not run._busy and build(run) == before
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
    assert evaluate(run, _validation()[:1]).example_count == 1


@pytest.mark.parametrize("kind", ("frozen_data", "moment_data", "resealed_cursor"))
def test_initial_training_serialization_partial_write_poisons_and_preserves_signal(
    kind: str,
) -> None:
    run = _short()
    run.step()
    row = run._rows[0]
    original_dump = type(row).model_dump
    changed = False

    def dump(current: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        if current is row and not changed:
            changed = True
            if kind == "frozen_data":
                next(run._backbone._model.parameters()).data.reshape(-1)[0].add_(0.125)
            elif kind == "moment_data":
                state = next(iter(run._optimizer._optimizer.state.values()))
                state["exp_avg"].data.reshape(-1)[0].add_(0.125)
            else:
                run._examples_seen += 1
                run._commit_owner()
            raise KeyboardInterrupt("primary serialization signal")
        return original_dump(current, *args, **kwargs)

    caller = capture_runtime_state(torch.device("cpu"))
    with (
        patch.object(type(row), "model_dump", new=dump),
        pytest.raises(KeyboardInterrupt, match="primary serialization signal"),
    ):
        evaluate(run, _validation()[:1])
    assert changed and run._poisoned and run._optimizer.poisoned and not run._busy
    assert _same_runtime(caller, capture_runtime_state(torch.device("cpu")))
