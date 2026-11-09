"""Actual C1/C5 kernel admission; test injection is NOT workflow/CLI dispatch."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import Qwen3ForCausalLM

from iris.application import _scientific_ablation_workflow_steps as steps
from iris.application import scientific_ablation_capture as capture_kernel
from iris.application import scientific_ablation_controlled_capture as controlled_capture_kernel
from iris.application import (
    scientific_ablation_controlled_evaluation_ledger as controlled_score_kernel,
)
from iris.application import scientific_ablation_evaluation_ledger as score_kernel
from iris.application.scientific_ablation_workflow import create_scientific_ablation_workflow_v1
from iris.artifacts.scientific_ablation_kernel_chunk_v1 import (
    ScientificAblationKernelChunkExpectationV1 as Chunk,
)
from iris.artifacts.scientific_ablation_workflow_v1 import scientific_ablation_workflow_phases_v1
from iris.utils.hashing import canonical_json_file_bytes
from tests.integration._workflow_geometry_setup import GEOMETRIES, Geometry, prepare
from tests.integration.test_scientific_ablation_execution import _tree

pytestmark = pytest.mark.integration


def _causes(error: BaseException) -> str:
    result = str(error)
    if error.__cause__ is not None:
        result += " " + _causes(error.__cause__)
    if isinstance(error, BaseExceptionGroup):
        result += " ".join(_causes(child) for child in error.exceptions)
    return result


@pytest.mark.parametrize("geometry", (GEOMETRIES[0], GEOMETRIES[2]), ids=("c1", "c5"))
def test_actual_kernel_exact_chunks_and_stale_retries_through_full_workflow(
    tmp_path: Path, geometry: Geometry
) -> None:
    case = prepare(tmp_path / "case", geometry)
    job = create_scientific_ablation_workflow_v1(
        root=tmp_path / "workflow", plan=case.plan, make_fresh_run=case.factory
    )
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def verify_stale(probe: dict[str, Any]) -> None:
        module, actual = probe["module"], probe["actual"]
        args, kwargs = probe["args"], probe["kwargs"]
        request, result, capture = probe["request"], probe["result"], probe["capture"]
        assert job._state == "idle"  # The ORIGINAL outer claim has exited.
        after = _tree(job._seal.path)
        primaries: list[BaseException] = []
        original_fail = module._fail

        def observed_failure(primary: BaseException | None, secondary: list[BaseException]) -> None:
            if primary is not None:
                primaries.append(primary)
            original_fail(primary, secondary)

        with (
            patch.object(module, "_fail", observed_failure),
            patch.object(module, "_stage", side_effect=AssertionError("No new stage")) as stage,
            patch.object(steps, "_owner", side_effect=AssertionError("No factory")) as factory,
            patch("torch.nn.Module.__init__", side_effect=AssertionError("No graph")) as graph,
            patch.object(
                Qwen3ForCausalLM, "forward", side_effect=AssertionError("No forward")
            ) as forward,
            pytest.raises(RuntimeError) as error,
        ):
            actual(*args, expected_chunk=request, **kwargs)
        assert any("Saved chunk prefix differs" in _causes(item) for item in primaries)
        if not capture:
            assert str(error.value) == "Mode-bound score ledger operation failed"
            assert error.value.__cause__ is None and error.value.__suppress_context__ is True
        assert stage.call_count == factory.call_count == graph.call_count == forward.call_count == 0
        assert _tree(job._seal.path) == after
        records.append(
            dict(
                namespace=module._STORE,
                request=request.model_dump(mode="json"),
                actual_acknowledged_rows=len(result.rows),
                actual_head_sha256=result.acknowledgements[-1].sha256,
                actual_complete_sha256=None if result.complete is None else result.complete.sha256,
                stale_retry_denied_before_stage_factory_graph_forward=True,
                stale_retry_preserved_all_workflow_bytes=True,
            )
        )

    def wrap(module: Any, name: str, capture: bool) -> Callable[..., Any]:
        actual = getattr(module, name)

        def guarded(*args: Any, **kwargs: Any) -> Any:
            assert kwargs["rows_per_call"] == 1 and "expected_chunk" not in kwargs
            digest = kwargs[
                "expected_plan_sha256" if capture else "expected_evaluation_plan_sha256"
            ]
            target = kwargs["root"] / module._STORE / digest
            if target.exists():
                parameters = (
                    (target, digest, kwargs["maximum_storage_bytes"])
                    if capture
                    else (
                        target,
                        digest,
                        kwargs["expected_plan_sha256"],
                        kwargs["expected_capture_sha256"],
                        kwargs["maximum_storage_bytes"],
                    )
                )
                previous = module._read(*parameters)
                start = len(previous.rows)
                head = previous.acknowledgements[-1].sha256
            else:
                start, head = 0, module._genesis(digest)
            assert start in (0, 1)  # The paired fixture has exactly two saved examples.
            request = Chunk(
                kernel_plan_sha256=digest,
                acknowledgement_head_sha256=head,
                acknowledged_rows=start,
                total_rows=2,
                rows_per_call=1,
                stop_row_index=start + 1,
            )
            result = actual(*args, expected_chunk=request, **kwargs)
            assert len(result.rows) == len(result.acknowledgements) == start + 1
            assert len(result.plan.payload.rows) == 2
            assert (result.complete is not None) is (start == 1)
            pending.append(
                dict(
                    module=module,
                    actual=actual,
                    args=args,
                    kwargs=dict(kwargs),
                    request=request,
                    result=result,
                    capture=capture,
                )
            )
            return result

        return guarded

    bindings = (
        (capture_kernel, "capture_scientific_ablation_heldout_v1", True),
        (controlled_capture_kernel, "capture_scientific_ablation_controlled_heldout_v1", True),
        (score_kernel, "evaluate_scientific_ablation_capture_resumable_v2", False),
        (
            controlled_score_kernel,
            "evaluate_scientific_ablation_controlled_capture_resumable_v1",
            False,
        ),
    )
    with ExitStack() as stack:
        candidate = stack.enter_context(
            patch(
                "iris.application.synthetic_scoring_v2.execute_python_job_v2",
                side_effect=AssertionError("No generated Python on host"),
            )
        )
        for module, name, capture in bindings:
            stack.enter_context(patch.object(module, name, wrap(module, name, capture)))
        for phase in scientific_ablation_workflow_phases_v1(case.plan):
            for _ in range(2 if phase.kind in ("capture", "score") else 1):
                progress = job.advance_expected(
                    expected_phase=phase,
                    expected_completed_phase_sha256=job.progress.completed_phase_sha256,
                    rows_per_call=1,
                )
                if phase.kind in ("capture", "score"):
                    assert len(pending) == 1
                    verify_stale(pending.pop())
                else:
                    assert not pending
        assert candidate.call_count == 0
    assert progress.complete and len(progress.completed_phase_sha256) == 10 and len(records) == 12
    (tmp_path / "actual-geometry-complete.json").write_bytes(
        canonical_json_file_bytes(
            asdict(geometry)
            | dict(
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                selected_step=1,
                terminal_step=3,
                real_original_native_workflow=True,
                source_scope="tiny_paired_geometry_augmentation_not_scientific5000",
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
    (tmp_path / "actual-kernel-chunk-expectation-witness.json").write_bytes(
        canonical_json_file_bytes(
            dict(
                kind="iris-kernel-chunk-expectation-native-witness-v1",
                plan_sha256=case.plan.sha256,
                completed_phase_sha256=progress.completed_phase_sha256,
                actual_kernel_requests=records,
                test_wrapper_injected_guard_not_public_workflow_dispatch=True,
                actual_os_cold_process=False,
                independently_persisted_pre_post_file_identity_snapshots=False,
                durable_cli_command_ack=False,
                full_installed_or_scientific_acceptance=False,
            )
        )
    )
