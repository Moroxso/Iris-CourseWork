"""Real original C1/C5 kernels/writers under a nonzero original-claim VIEW.

The test supplies the private reservation view at the existing claim seam, as a
future trusted coordinator must. All kernel/native results are original calls,
not injected outputs. This is NOT a durable journal, cold command recovery,
global disk lease, installed delivery or scientific/main-model evidence.
"""

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_workflow as workflow
from iris.application.scientific_ablation_workflow_io import (
    IrisScientificAblationWorkflowStorageError,
    ScientificAblationWorkflowStoreV1,
    _Claim,
    _tree_size,
)
from iris.application.scientific_ablation_writer_io import (
    write_scientific_ablation_writer_bundle_v1,
)
from iris.artifacts.scientific_ablation_workflow_kernel_v1 import (
    IrisScientificAblationWorkflowKernelProgressV1 as Kernel,
)
from iris.artifacts.scientific_ablation_workflow_v1 import (
    IrisScientificAblationWorkflowPlanV1,
    ScientificAblationWorkflowPhaseV1,
    scientific_ablation_workflow_phases_v1,
)
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, GeometryCase, prepare
from tests.integration.test_scientific_ablation_execution import _tree
from tests.integration.test_scientific_ablation_workflow_producer_result import (
    _call,
    _committed,
    _native_commit_retry,
    _next,
    _wire,
)
from tests.integration.test_scientific_ablation_workflow_public_chunk import (
    _binding,
    _first_request,
)

pytestmark = pytest.mark.integration
HOLD = 4 * 1024**2
METADATA = 4 * 1024**2


@pytest.fixture(scope="module", params=(GEOMETRIES[0], GEOMETRIES[2]), ids=("c1", "c5"))
def native_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> GeometryCase:
    return prepare(tmp_path_factory.mktemp("reservation-original-source") / "case", request.param)


def test_real_four_kernels_native_retry_and_writer_publication_keep_the_reserved_view(
    native_case: GeometryCase, tmp_path: Path
) -> None:
    case = native_case
    before_source = _tree(case.root)
    root = tmp_path / "workflow"
    job = workflow.create_scientific_ablation_workflow_v1(
        root=root, plan=case.plan, make_fresh_run=case.factory
    )
    original_claim = ScientificAblationWorkflowStoreV1.claim
    original_budget, original_native = steps.budget, steps._native_reservation
    original_publish = write_scientific_ablation_writer_bundle_v1
    claims: list[dict[str, Any]] = []
    budgets: list[dict[str, Any]] = []
    native_sizes: list[dict[str, Any]] = []
    returns: list[dict[str, Any]] = []

    @contextmanager
    def claimed(store: ScientificAblationWorkflowStoreV1) -> Iterator[_Claim]:
        assert store.path == job._seal.path
        with original_claim(store) as base:
            held = base.with_reserved_bytes(HOLD)
            assert held.lock is base.lock and held.identity == base.identity
            assert held.reserved_bytes == HOLD and base.reserved_bytes == 0
            metadata = os.fstat(held.lock.descriptor)
            claims.append(
                dict(
                    root=str(store.path),
                    held_bytes=HOLD,
                    lock_device=metadata.st_dev,
                    lock_inode=metadata.st_ino,
                )
            )
            yield held

    def budget(
        store: ScientificAblationWorkflowStoreV1,
        subtree: Path,
        claim: _Claim,
        reserve: int,
        *,
        kernel: bool = True,
    ) -> int:
        assert claim.reserved_bytes == HOLD
        occupied = _tree_size(store.path)[0]
        previous = _tree_size(subtree)[0] if subtree.exists() else 0
        available = (
            case.plan.payload.resources.maximum_storage_bytes
            - (occupied - previous)
            - reserve
            - HOLD
        )
        expected = (
            min(case.plan.payload.resources.maximum_kernel_storage_bytes, available)
            if kernel
            else available
        )
        actual = original_budget(store, subtree, claim, reserve, kernel=kernel)
        assert actual == expected and claim.minimum_free_for(reserve) == reserve + HOLD
        budgets.append(
            dict(
                subtree=str(subtree),
                occupied=occupied,
                previous_subtree_bytes=previous,
                reserve=reserve,
                held_bytes=HOLD,
                kernel=kernel,
                quota=actual,
            )
        )
        return actual

    def native_size(
        plan: IrisScientificAblationWorkflowPlanV1,
        phase: ScientificAblationWorkflowPhaseV1,
    ) -> tuple[int, bool]:
        result = original_native(plan, phase)
        native_sizes.append(
            dict(
                phase=phase.name, kind=phase.kind, reserved_bytes=result[0], no_next_cycle=result[1]
            )
        )
        return result

    # Observation delegates unchanged arguments/results to the original functions.
    kernel_calls: dict[str, Any] = {}
    with ExitStack() as stack:
        stack.enter_context(patch.object(ScientificAblationWorkflowStoreV1, "claim", new=claimed))
        stack.enter_context(patch.object(steps, "budget", new=budget))
        stack.enter_context(patch.object(steps, "_native_reservation", new=native_size))
        published = stack.enter_context(
            patch.object(
                steps, "write_scientific_ablation_writer_bundle_v1", wraps=original_publish
            )
        )
        candidate = stack.enter_context(
            patch(
                "iris.application.synthetic_scoring_v2.execute_python_job_v2",
                side_effect=AssertionError("No generated Python on host"),
            )
        )
        phases = scientific_ablation_workflow_phases_v1(case.plan)
        for phase in phases:
            if phase.kind not in ("capture", "score"):
                result = (
                    _native_commit_retry(job, phase, root)
                    if phase.kind == "diagnostics"
                    else _call(job, phase)
                )
                _committed(job, phase, result)
                returns.append(_wire(result))
                continue
            module, _, operation = _binding(phase)
            if operation not in kernel_calls:
                kernel_calls[operation] = stack.enter_context(
                    patch.object(module, operation, wraps=getattr(module, operation))
                )
            request = _first_request(job, phase)
            for row in range(2):
                result = _call(job, phase, request)
                assert result.kernel_progress_json is not None
                observed = Kernel.model_validate_json(result.kernel_progress_json)
                assert observed.acknowledged_rows == row + 1 and observed.total_rows == 2
                # Exact original kernel return precedes workflow phase commit;
                # it is NOT a later inspection of the now-advanced phase head.
                assert not observed.phase_is_acknowledged
                assert observed.completed_phase_sha256 == result.completed_phase_sha256_before
                assert (observed.kernel_complete_sha256 is not None) is (row == 1)
                assert result.expected_chunk_json == steps.encoded(request)
                if row == 1:
                    _committed(job, phase, result)
                returns.append(_wire(result))
                if row == 0:
                    request = _next(request, observed)
        assert candidate.call_count == 0
    assert len(phases) == 10 and len(returns) == 16 and len(kernel_calls) == 4
    assert sorted(call.call_count for call in kernel_calls.values()) == [2, 2, 4, 4]
    for call in kernel_calls.values():
        for invocation in call.call_args_list:
            assert invocation.kwargs["minimum_free_bytes"] == METADATA + HOLD
            assert invocation.kwargs["rows_per_call"] == 1
    assert published.call_count == (0 if case.geometry.cycles == 1 else 2)
    publications = []
    for invocation in published.call_args_list:
        assert invocation.kwargs["minimum_free_bytes"] == METADATA + HOLD
        publications.append(
            dict(
                protocol=invocation.kwargs["expected_protocol"],
                root=str(invocation.kwargs["root"]),
                maximum_storage_bytes=invocation.kwargs["maximum_storage_bytes"],
                minimum_free_bytes=invocation.kwargs["minimum_free_bytes"],
                report_sha256=invocation.kwargs["expected_report_sha256"],
                source_sha256=invocation.kwargs["expected_source_sha256"],
            )
        )
    assert len(native_sizes) == 3
    assert [item["kind"] for item in native_sizes] == ["diagnostics", "workspace", "message"]
    if case.geometry.cycles == 1:
        assert all(
            item["reserved_bytes"] == METADATA and item["no_next_cycle"]
            for item in native_sizes
            if item["kind"] in ("workspace", "message")
        )
    assert job.progress.complete and _tree(case.root) == before_source
    (tmp_path / "actual-reservation-native-workflow-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-original-reservation-native-workflow-witness-v1",
                geometry=asdict(case.geometry),
                source_root=str(case.root),
                workflow_root=str(root),
                plan=case.plan.model_dump(mode="json"),
                completed_phase_sha256=job.progress.completed_phase_sha256,
                held_bytes=HOLD,
                claims=claims,
                budgets=budgets,
                native_reservations=native_sizes,
                publications=publications,
                original_returns=returns,
                original_claim_view_supplied_at_test_seam=True,
                kernel_arguments_and_results_observed_not_rewritten=True,
                separate_one_row_requests_not_durable_multirow_recovery=True,
                diagnostic_original_same_owner_storage_retry=True,
                cold_handle_native_uncertainty_not_remeasurement=True,
                independent_durable_producer_result=False,
                installed_or_scientific_acceptance=False,
            )
        )
    )


def test_real_aggregate_hold_rejects_capture_before_factory_graph_or_row_stage(
    native_case: GeometryCase, tmp_path: Path
) -> None:
    case = native_case
    job = workflow.create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow", plan=case.plan, make_fresh_run=case.factory
    )
    original_claim = ScientificAblationWorkflowStoreV1.claim
    observations = []

    @contextmanager
    def pressured(store: ScientificAblationWorkflowStoreV1) -> Iterator[_Claim]:
        with original_claim(store) as base:
            occupied = _tree_size(store.path)[0]
            held_bytes = (
                store.plan.payload.resources.maximum_storage_bytes - occupied - METADATA // 2
            )
            held = base.with_reserved_bytes(held_bytes)
            observations.append(
                dict(
                    occupied=occupied,
                    held_bytes=held_bytes,
                    maximum_storage_bytes=store.plan.payload.resources.maximum_storage_bytes,
                )
            )
            yield held

    phase = scientific_ablation_workflow_phases_v1(case.plan)[0]
    with (
        patch.object(ScientificAblationWorkflowStoreV1, "claim", new=pressured),
        patch.object(steps, "_owner", side_effect=AssertionError("No capture factory")) as factory,
        patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
        patch.object(
            Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
        ) as forward,
        pytest.raises(IrisScientificAblationWorkflowStorageError) as raised,
    ):
        job.advance_expected(
            expected_phase=phase, expected_completed_phase_sha256=(), rows_per_call=1
        )
    assert isinstance(raised.value.__cause__, ValueError)
    assert factory.call_count == graph.call_count == forward.call_count == 0
    assert observations and job.progress.completed_phase_sha256 == () and job._state == "idle"
    assert not (job._seal.path / "kernels").exists()
    (tmp_path / "actual-reservation-quota-denial-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-original-reservation-aggregate-denial-witness-v1",
                geometry=asdict(case.geometry),
                workflow_root=str(tmp_path / "workflow"),
                plan=case.plan.model_dump(mode="json"),
                reservations=observations,
                original_factory_graph_forward_row_stage_calls=0,
                actual_aggregate_quota_not_simulated_disk_full=True,
                independent_durable_producer_result=False,
                installed_or_scientific_acceptance=False,
            )
        )
    )
