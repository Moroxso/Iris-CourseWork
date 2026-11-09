"""Real local quota/lock checks and labelled kernel-boundary probes; no native ACK."""

import shutil
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import PropertyMock, patch

import pytest

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture
from iris.application import scientific_ablation_controlled_capture as controlled_capture
from iris.application import scientific_ablation_controlled_evaluation_ledger as controlled_score
from iris.application import scientific_ablation_evaluation_ledger as score
from iris.application import scientific_ablation_workflow_io as storage
from iris.application.scientific_ablation_controlled_heldout import (
    IrisScientificAblationControlledHeldoutV1,
)
from iris.application.scientific_ablation_heldout import IrisScientificAblationHeldoutV1
from iris.artifacts.scientific_ablation_workflow_v1 import (
    ScientificAblationWorkflowIntentV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.interventions.policy import WorkspaceInterventionPlanV1
from tests.unit._workflow_data_setup import intent


def store_at(path: Path, *, minimum: int = 0) -> storage.ScientificAblationWorkflowStoreV1:
    data = intent(path, (WorkspaceInterventionPlanV1(interventions=()),)).model_dump(mode="python")
    data["resources"]["minimum_free_bytes"] = minimum
    plan = storage.build_scientific_ablation_workflow_plan_v1(
        ScientificAblationWorkflowIntentV1.model_validate(data)
    )
    return storage.ScientificAblationWorkflowStoreV1(path / "output", plan)


def test_reservation_view_is_immutable_additive_same_live_lock_without_allocations(
    tmp_path: Path,
) -> None:
    store = store_at(tmp_path)

    def snapshot() -> dict[str, tuple[int, int, int, int, int]]:
        observed = {}
        for path in store.path.iterdir():
            metadata = path.lstat()
            observed[path.name] = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        return observed

    with store.claim() as base:
        # Windows correctly denies a second read of the held lock byte. Check
        # its metadata while held and its contents only after release.
        before = snapshot()
        plan_bytes = (store.path / "plan.json").read_bytes()
        held = base.with_reserved_bytes(1234)
        nested = held.with_reserved_bytes(5678)
        assert base.reserved_bytes == 0 and held.reserved_bytes == 1234
        assert nested.reserved_bytes == 6912
        assert nested.store is held.store is base.store
        assert nested.lock is held.lock is base.lock and nested.identity == base.identity
        assert nested.minimum_free_for(10) == 6922
        with pytest.raises(FrozenInstanceError):
            unchecked: Any = nested
            unchecked.reserved_bytes = 0
        assert snapshot() == before
        assert (store.path / "plan.json").read_bytes() == plan_bytes
        base.lock.check()
    assert (store.path / ".iris-owner.lock").read_bytes() == b"1"
    with pytest.raises(OSError):
        nested.reserve(0)  # It does not survive release of its original OS descriptor.


@pytest.mark.parametrize("amount", (True, False, -1, 1.0, "1", None))
def test_invalid_additional_reservation_never_coerced(tmp_path: Path, amount: Any) -> None:
    store = store_at(tmp_path)
    with store.claim() as base, pytest.raises(ValueError):
        base.with_reserved_bytes(amount)


@pytest.mark.parametrize("amount", (True, False, -1, 1.0, "1", None))
def test_invalid_kernel_free_floor_never_coerced(tmp_path: Path, amount: Any) -> None:
    store = store_at(tmp_path)
    with store.claim() as base, pytest.raises(ValueError):
        base.minimum_free_for(amount)


@pytest.mark.parametrize("amount", (True, -1, 1.0))
def test_forged_reservation_field_is_not_accepted(tmp_path: Path, amount: Any) -> None:
    store = store_at(tmp_path)
    with store.claim() as base, pytest.raises(ValueError):
        replace(base, reserved_bytes=amount).check()


def test_view_holds_the_remaining_whole_quota_but_is_not_a_global_lease(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    with store.claim() as base:
        remaining = (
            store.plan.payload.resources.maximum_storage_bytes - storage._tree_size(store.path)[0]
        )
        with pytest.raises(ValueError):
            base.with_reserved_bytes(remaining + 1)
        held = base.with_reserved_bytes(remaining)
        held.reserve(0)
        with pytest.raises(ValueError):
            held.reserve(1)
        # Contract is an explicit immutable VIEW: the future coordinator must pass
        # it through every producer path, not retain/use the unreserved original.
        base.reserve(1)


@pytest.mark.parametrize("kernel", (False, True))
def test_kernel_or_publication_budget_excludes_held_result_copy_and_counts_siblings(
    tmp_path: Path, kernel: bool
) -> None:
    store = store_at(tmp_path)
    with store.claim() as base:
        target = store.path / "kernels/ordinary-capture"
        sibling = store.path / "kernels/ordinary-score"
        target.mkdir(parents=True)
        sibling.mkdir()
        (target / "owned-fixture.bin").write_bytes(b"target" * 100)
        (sibling / "owned-fixture.bin").write_bytes(b"sibling" * 100)
        metadata, held_bytes = 4096, 123456
        original = steps.budget(store, target, base, metadata, kernel=kernel)
        held = base.with_reserved_bytes(held_bytes)
        reduced = steps.budget(store, target, held, metadata, kernel=kernel)
        assert reduced == original - held_bytes
        assert held.minimum_free_for(metadata) == held_bytes + metadata
        occupied = storage._tree_size(store.path)[0]
        subtree = storage._tree_size(target)[0]
        assert (
            reduced
            == store.plan.payload.resources.maximum_storage_bytes
            - (occupied - subtree)
            - metadata
            - held_bytes
        )


def test_disk_floor_counts_held_bytes_without_changing_the_plan(tmp_path: Path) -> None:
    store = store_at(tmp_path, minimum=4096)
    original_plan = store.plan.model_dump_json()
    with store.claim() as base:
        held = base.with_reserved_bytes(1024)
        with patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=5120)):
            held.reserve(0)
            with pytest.raises(ValueError):
                held.reserve(1)
        assert held.minimum_free_for(17) == 5137
    assert store.plan.model_dump_json() == original_plan


def test_kernel_free_floor_over_existing_limit_rejects_before_factory(tmp_path: Path) -> None:
    store = store_at(tmp_path, minimum=16 * 1024**3)
    phase = scientific_ablation_workflow_phases_v1(store.plan)[0]
    with (
        patch.object(shutil, "disk_usage", return_value=SimpleNamespace(free=64 * 1024**3)),
        store.claim() as base,
        patch.object(
            steps, "_owner", side_effect=AssertionError("No graph before budget")
        ) as factory,
        pytest.raises(ValueError),
    ):
        steps.execute(store, phase, (), None, base, 1)
    assert factory.call_count == 0


@pytest.mark.parametrize("kind", ("workspace", "message"))
def test_C1_existing_four_MiB_reservation_not_inflated_or_bypassed(
    tmp_path: Path, kind: str
) -> None:
    store = store_at(tmp_path)
    phase = next(p for p in scientific_ablation_workflow_phases_v1(store.plan) if p.kind == kind)
    labelled_source = SimpleNamespace(
        inference=SimpleNamespace(
            manifest=SimpleNamespace(
                payload=SimpleNamespace(model=SimpleNamespace(workspace=SimpleNamespace(cycles=1)))
            )
        )
    )
    with patch.object(steps, "source", return_value=labelled_source):
        assert steps._native_reservation(store.plan, phase) == (4 * 1024**2, True)
    with store.claim() as base:
        held = base.with_reserved_bytes(256)
        held.reserve(4 * 1024**2)
        assert held.minimum_free_for(4 * 1024**2) == 4 * 1024**2 + 256


@pytest.mark.parametrize("kind", ("capture", "score"))
@pytest.mark.parametrize("controlled", (False, True))
def test_all_four_original_kernel_calls_receive_the_derived_free_floor(
    tmp_path: Path, kind: str, controlled: bool
) -> None:
    """Labelled abort at the real call boundary, NOT a native/row-ACK result."""
    store = store_at(tmp_path)
    phases = scientific_ablation_workflow_phases_v1(store.plan)
    phase = phases[(2 if controlled else 0) + (1 if kind == "score" else 0)]
    labelled_prefix: Any = tuple(
        SimpleNamespace(record=SimpleNamespace(sha256="a" * 64)) for _ in range(phase.index)
    )
    owner_type = (
        IrisScientificAblationControlledHeldoutV1 if controlled else IrisScientificAblationHeldoutV1
    )
    module: Any = (
        (controlled_capture if controlled else capture)
        if kind == "capture"
        else (controlled_score if controlled else score)
    )
    builder = (
        (
            "build_scientific_ablation_controlled_capture_plan_v1"
            if controlled
            else "build_scientific_ablation_capture_plan_v1"
        )
        if kind == "capture"
        else (
            "build_scientific_ablation_controlled_evaluation_plan_v1"
            if controlled
            else "build_scientific_ablation_evaluation_plan_v2"
        )
    )
    operation = (
        (
            "capture_scientific_ablation_controlled_heldout_v1"
            if controlled
            else "capture_scientific_ablation_heldout_v1"
        )
        if kind == "capture"
        else (
            "evaluate_scientific_ablation_controlled_capture_resumable_v1"
            if controlled
            else "evaluate_scientific_ablation_capture_resumable_v2"
        )
    )
    stopped = RuntimeError("Labelled call boundary, no kernel execution")
    with store.claim() as base, ExitStack() as stack:
        held = base.with_reserved_bytes(32768)
        stack.enter_context(patch.object(steps, "_owner", return_value=object.__new__(owner_type)))
        stack.enter_context(patch.object(steps, "_score_args", return_value={}))
        stack.enter_context(
            patch.object(
                owner_type,
                "manifest",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(sha256="b" * 64),
            )
        )
        stack.enter_context(
            patch.object(module, builder, return_value=SimpleNamespace(sha256="c" * 64))
        )
        reached = stack.enter_context(patch.object(module, operation, side_effect=stopped))
        with pytest.raises(RuntimeError) as raised:
            steps.execute(store, phase, labelled_prefix, None, held, 1)
        assert raised.value is stopped and reached.call_count == 1
        assert reached.call_args.kwargs["minimum_free_bytes"] == 4 * 1024**2 + 32768


@pytest.mark.parametrize("kind", ("workspace", "message"))
def test_both_writer_publication_calls_receive_the_derived_free_floor(
    tmp_path: Path, kind: str
) -> None:
    """Labelled abort at the original writer boundary; no published/native DATA."""
    store = store_at(tmp_path)
    phase = next(
        p for p in scientific_ablation_workflow_phases_v1(store.plan) if p.kind == "publish"
    )
    labelled_data = SimpleNamespace(
        record=SimpleNamespace(
            sha256="a" * 64,
            payload=SimpleNamespace(
                phase=SimpleNamespace(kind=kind, index=0), origin="original_native_result_data"
            ),
        ),
        files=(
            ("source.json", b"source"),
            ("report.json", b"report"),
            ("runtime.json", b"runtime"),
        ),
    )
    labelled_prefix: Any = (labelled_data,)
    labelled_writer = (
        SimpleNamespace(sha256="c" * 64, payload=SimpleNamespace(maximum_report_bytes=1024**2)),
        SimpleNamespace(sha256="b" * 64),
        SimpleNamespace(),
    )
    stopped = RuntimeError("Labelled publication boundary, no writer execution")
    with store.claim() as base:
        held = base.with_reserved_bytes(32768)
        expected_quota = steps.budget(
            store, store.path / "published" / kind, held, 4 * 1024**2, kernel=False
        )
        with (
            patch.object(steps, "_writer", return_value=labelled_writer),
            patch.object(
                steps, "write_scientific_ablation_writer_bundle_v1", side_effect=stopped
            ) as write,
            pytest.raises(RuntimeError) as raised,
        ):
            steps.execute(store, phase, labelled_prefix, None, held, None)
        assert raised.value is stopped and write.call_count == 1
        assert write.call_args.kwargs["minimum_free_bytes"] == 4 * 1024**2 + 32768
        assert write.call_args.kwargs["maximum_storage_bytes"] == expected_quota
        assert not (store.path / "published").exists()


@pytest.mark.parametrize("fault", (KeyboardInterrupt(), SystemExit(17)))
def test_quota_inventory_signal_preserves_exact_primary_object(
    tmp_path: Path, fault: BaseException
) -> None:
    store = store_at(tmp_path)
    with pytest.raises(type(fault)) as raised, store.claim() as base:
        held = base.with_reserved_bytes(1024)
        with patch.object(storage, "_tree_size", side_effect=fault):
            held.reserve(0)
    assert raised.value is fault
