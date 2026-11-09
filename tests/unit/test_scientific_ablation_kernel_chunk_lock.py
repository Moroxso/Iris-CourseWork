"""Real OS locks plus explicitly mocked plans/row DATA; no native acceptance."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)

KERNELS = (
    "scientific_ablation_capture",
    "scientific_ablation_controlled_capture",
    "scientific_ablation_evaluation_ledger",
    "scientific_ablation_controlled_evaluation_ledger",
)
PLAN, HEAD = "a" * 64, "b" * 64


class _ObservedWithoutNativeWork(RuntimeError):
    pass


def invoke(module: Any, root: Path, chunk: Chunk, heldout: Any, budget: int = 1) -> None:
    arguments: dict[str, Any] = dict(
        root=root, rows_per_call=budget, expected_chunk=chunk, minimum_free_bytes=0
    )
    if "capture" in module.__name__:
        name = (
            "capture_scientific_ablation_controlled_heldout_v1"
            if "controlled" in module.__name__
            else "capture_scientific_ablation_heldout_v1"
        )
        getattr(module, name)(
            heldout, expected_input_sha256=PLAN, expected_plan_sha256=PLAN, **arguments
        )
    else:
        name = (
            "evaluate_scientific_ablation_controlled_capture_resumable_v1"
            if "controlled" in module.__name__
            else "evaluate_scientific_ablation_capture_resumable_v2"
        )
        getattr(module, name)(
            expected_evaluation_plan_sha256=PLAN,
            capture_path=root / "unused",
            expected_plan_sha256=PLAN,
            expected_capture_sha256=PLAN,
            lifecycle_path=root / "unused",
            inference_path=root / "unused",
            inputs_root=root / "unused",
            training=None,
            validation=None,
            test=None,
            python_policy=None,
            **arguments,
        )


def causes(error: BaseException) -> str:
    result = str(error)
    if error.__cause__ is not None:
        result += " " + causes(error.__cause__)
    if isinstance(error, BaseExceptionGroup):
        result += " ".join(causes(child) for child in error.exceptions)
    return result


def observe_failure(module: Any, primaries: list[BaseException]) -> Any:
    original = module._fail

    def observed(primary: BaseException | None, secondary: list[BaseException]) -> None:
        if primary is not None:
            primaries.append(primary)
        original(primary, secondary)

    return observed


def check_public_error(module: Any, error: BaseException) -> None:
    if "evaluation_ledger" in module.__name__:
        assert str(error) == "Mode-bound score ledger operation failed"
        assert error.__cause__ is None and error.__suppress_context__ is True


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize(
    "scenario", ("fresh_stale", "existing_stale", "existing_exact", "lock_race")
)
def test_guard_uses_actual_prefix_under_real_kernel_lock(
    tmp_path: Path, kernel: str, scenario: str
) -> None:
    module: Any = import_module("iris.application." + kernel)
    root = tmp_path / "root"
    target = root / module._STORE / PLAN
    fresh = scenario == "fresh_stale"
    if not fresh:
        target.mkdir(parents=True)
    plan = SimpleNamespace(sha256=PLAN, payload=SimpleNamespace(rows=(0, 1)))
    progress = SimpleNamespace(
        plan=plan, rows=(object(),), acknowledgements=(SimpleNamespace(sha256=HEAD),)
    )
    heldout = Mock()
    inputs = SimpleNamespace(
        rows=(0, 1), capture=SimpleNamespace(path=root / "unused"), check=Mock()
    )
    original_lock, original_guard = module.process_file_lock, module.check_kernel_chunk_prefix_v1
    held: list[Any] = []
    observations: list[tuple[int, str]] = []
    primaries: list[BaseException] = []

    @contextmanager
    def lock_scope(path: Path) -> Iterator[Any]:
        with original_lock(path) as lock:
            held.append(lock)
            if scenario == "lock_race":
                # Explicitly mocked external commit appears only after admission.
                progress.rows = (object(), object())
                progress.acknowledgements = (SimpleNamespace(sha256="c" * 64),)
            try:
                yield lock
            finally:
                held.pop()

    def guard(value: Chunk | None, **actual: Any) -> None:
        assert len(held) == 1
        held[0].check()
        observations.append((actual["acknowledged_rows"], actual["acknowledgement_head_sha256"]))
        original_guard(value, **actual)
        # A passing admission probe still performs no row/stage/native work.
        raise _ObservedWithoutNativeWork("exact locked prefix observed")

    chunk = Chunk(
        kernel_plan_sha256=PLAN,
        acknowledgement_head_sha256=HEAD if scenario != "existing_stale" else "d" * 64,
        acknowledged_rows=1,
        total_rows=2,
        rows_per_call=1,
        stop_row_index=2,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "_fail", observe_failure(module, primaries)))
        stack.enter_context(patch.object(module, "process_file_lock", lock_scope))
        stack.enter_context(patch.object(module, "check_kernel_chunk_prefix_v1", guard))
        reader = stack.enter_context(patch.object(module, "_read", return_value=progress))
        stage = stack.enter_context(patch.object(module, "_stage"))
        if "capture" in kernel:
            builder = (
                "build_scientific_ablation_controlled_capture_plan_v1"
                if "controlled" in kernel
                else "build_scientific_ablation_capture_plan_v1"
            )
            stack.enter_context(patch.object(module, builder, return_value=plan))
        else:
            stack.enter_context(patch.object(module, "_prepare", return_value=inputs))
            stack.enter_context(patch.object(module, "_plan", return_value=plan))
            stack.enter_context(patch.object(module, "_capture_stamp", return_value={}))
        with pytest.raises(RuntimeError) as error:
            invoke(module, root, chunk, heldout)
        if scenario == "existing_exact":
            assert any("exact locked prefix observed" in causes(item) for item in primaries)
        else:
            assert any("Saved chunk prefix differs" in causes(item) for item in primaries)
        check_public_error(module, error.value)
        assert stage.call_count == 0 and heldout.execute_row.call_count == 0
        assert reader.call_count == (0 if fresh else 1)
    assert not held and len(observations) == 1
    if scenario == "lock_race":
        assert observations == [(2, "c" * 64)]
    if fresh:
        assert not target.exists()  # No plan, attempt or stage on stale genesis intent.
    else:
        assert list(target.iterdir()) == []


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("mismatch", ("budget", "plan"))
def test_saved_intent_mismatch_fails_before_inputs_lock_or_directories(
    tmp_path: Path, kernel: str, mismatch: str
) -> None:
    module: Any = import_module("iris.application." + kernel)
    chunk = Chunk(
        kernel_plan_sha256="c" * 64 if mismatch == "plan" else PLAN,
        acknowledgement_head_sha256=HEAD,
        acknowledged_rows=0,
        total_rows=2,
        rows_per_call=1,
        stop_row_index=1,
    )
    root = tmp_path / "must-not-exist"
    primaries: list[BaseException] = []
    with (
        patch.object(module, "_fail", observe_failure(module, primaries)),
        patch.object(module, "process_file_lock") as lock,
        patch.object(module, "_stage") as stage,
        pytest.raises(RuntimeError) as error,
    ):
        invoke(module, root, chunk, Mock(), budget=2 if mismatch == "budget" else 1)
    assert any("saved kernel plan and chunk budget" in causes(item) for item in primaries)
    check_public_error(module, error.value)
    assert lock.call_count == stage.call_count == 0 and not root.exists()
